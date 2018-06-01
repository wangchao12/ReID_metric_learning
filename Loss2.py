import torch
from SummaryWriter import SummaryWriter
debuger = SummaryWriter('debuger.mat')


def all_diffs(a, b):
    """
    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).
    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    """
    return torch.unsqueeze(input=a, dim=1) - torch.unsqueeze(input=b, dim=0)


def cdist(a, b):
    """
    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.
    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2)
    """
    diffs = all_diffs(a, b)
    return torch.norm(diffs, 2, -1)

def metric_loss(fc, batch_person, num_fc):
    person_center = torch.unsqueeze(torch.mean(input=fc, dim=1), 1)
    center_diff = torch.norm(fc - person_center, dim=-1)
    center_loss = torch.mean(center_diff)
    cross_center = person_center.view(1, -1, num_fc)
    cross_center = cross_center.expand(batch_person, batch_person, num_fc)
    cross_diff = torch.norm(person_center - cross_center, dim=-1)
    values, idx = torch.topk(-cross_diff.view(-1, batch_person*batch_person), batch_person*2)
    cross_loss = torch.sum(-values) / batch_person
    return center_loss, cross_loss, center_loss/cross_loss


def metric_loss2(fc):
    person_center = torch.unsqueeze(torch.mean(input=fc, dim=1), 1)
    center_diff = torch.norm(fc - person_center, dim=-1)
    center_loss = torch.mean(center_diff)
    values, idx = torch.min(center_diff, -1)
    cross_loss = torch.mean(values)
    return center_loss, cross_loss, center_loss/cross_loss


def metric_loss3(fc, batch_person, num_file, fcs):
    fc = fc.view(batch_person, num_file, fcs)
    person_center = torch.unsqueeze(torch.mean(input=fc, dim=1), 1)
    center_diff = torch.norm(fc - person_center, dim=-1)
    center_loss_mean = torch.mean(center_diff)
    max_center, idx = torch.max(center_diff, -1)
    center_loss_max = torch.mean(max_center)
    center_loss = 0.5 * (center_loss_max + center_loss_mean)
    reshape_center = person_center.view(1, batch_person, -1)
    cross_centers = reshape_center.expand(batch_person, batch_person, fcs)
    cross_loss = 0
    for i in range(batch_person):
        cross_i = torch.unsqueeze(cross_centers[:, i, :], 1)
        cross_diss = torch.norm(fc - cross_i, dim=-1)
        if i == 0:
            cross_diss_ = cross_diss[1:batch_person]
        elif i == batch_person - 1:
            cross_diss_ = cross_diss[0:batch_person - 1]
        else:
            cross_diss_ = torch.cat((cross_diss[0: i], cross_diss[i + 1: batch_person]), 0)
        cross_diss_ = cross_diss_.view(-1, (batch_person-1)*num_file)
        values, idx = torch.min(cross_diss_, -1)
        cross_loss += values
    cross_loss = cross_loss / batch_person
    return center_loss, cross_loss, center_loss / cross_loss



def CenterEasyLoss(fc, batch_person, num_file, fcs):
    fc = fc.view(batch_person, num_file, fcs)
    person_center = torch.mean(input=fc, dim=1)
    center_loss = torch.mean(torch.norm(fc - torch.unsqueeze(person_center, dim=1), dim=-1))
    cross_matrix = cdist(person_center, person_center)
    cross_mask = torch.where(cross_matrix > torch.zeros_like(cross_matrix),   #把相同元素距离去掉，去除零
                             torch.ones_like(cross_matrix), torch.zeros_like(cross_matrix))
    cross_vector = torch.masked_select(cross_matrix, cross_mask.byte())
    min_cross, idx = torch.min(cross_vector.view(batch_person, -1), -1)       #其它类中心与目标最小距离
    cross_loss = torch.mean(min_cross)

    return center_loss, cross_loss, center_loss / cross_loss