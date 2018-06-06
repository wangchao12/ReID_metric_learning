import torch as th
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
    return th.unsqueeze(input=a, dim=1) - th.unsqueeze(input=b, dim=0)


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
    return th.norm(diffs, 2, -1)


def CenterEasyLoss(fc, batch_person, num_file, margin, fcs):
    fc = fc.view(batch_person, num_file, fcs)
    person_center = th.mean(input=fc, dim=1)
    center_loss = th.mean(th.norm(fc - th.unsqueeze(person_center, dim=1), dim=-1))
    cross_matrix = cdist(person_center, person_center)
    cross_mask = th.ones_like(cross_matrix) - th.eye(batch_person).to('cuda')
    cross_vector = th.masked_select(cross_matrix, cross_mask.byte())
    # print('size', cross_vector.size())
    min_cross, idx = th.min(cross_vector.view(batch_person, -1), -1)       #其它类中心与目标最小距离
    cross_loss = th.mean(min_cross)

    return center_loss, cross_loss, center_loss / cross_loss



def CenterEasyLoss2(fc, batch_person, num_file, fcs):
    fc = fc.view(batch_person, num_file, fcs)
    person_center = th.mean(input=fc, dim=1)
    distance = th.norm(fc - th.unsqueeze(person_center, dim=1), dim=-1)
    center_mean_loss = th.mean(distance)
    mask = th.where(distance > center_mean_loss, th.ones_like(distance), th.zeros_like(distance))
    center_up = th.masked_select(distance, mask.to('cuda').byte())
    center_loss = th.mean(center_up)
    cross_matrix = cdist(person_center, person_center)
    cross_mask = th.ones_like(cross_matrix) - th.eye(batch_person).to('cuda')
    cross_vector = th.masked_select(cross_matrix, cross_mask.byte())
    # print('size', cross_vector.size())
    min_cross, idx = th.min(cross_vector.view(batch_person, -1), -1)       #其它类中心与目标最小距离
    cross_loss = th.mean(min_cross)

    return center_loss, cross_loss, center_loss / cross_loss



def CenterEasyLoss3(fc, batch_person, num_file, fcs):
    fc = fc.view(batch_person, num_file, fcs)
    person_center = th.mean(input=fc, dim=1)
    distance = th.norm(fc - th.unsqueeze(person_center, dim=1), p=2, dim=-1)
    (center_max, idx) = th.max(distance, dim=-1)
    center_mean_loss = th.mean(distance)
    center_max_loss = th.mean(center_max)
    center_loss = 0.5 * (center_max_loss + center_mean_loss)
    cross_matrix = cdist(person_center, person_center)
    cross_mask = th.ones_like(cross_matrix) - th.eye(batch_person).to('cuda')
    cross_vector = th.masked_select(cross_matrix, cross_mask.byte())
    # print('size', cross_vector.size())
    min_cross, idx = th.min(cross_vector.view(batch_person, -1), -1)       #其它类中心与目标最小距离
    cross_loss = th.mean(min_cross)

    return center_loss, cross_loss, center_loss / cross_loss


def CenterEasyLoss4(fc, batch_person, num_file, fcs):
    fc = fc.view(batch_person, num_file, fcs)
    person_center = th.mean(input=fc, dim=1)
    distance = th.norm(fc - th.unsqueeze(person_center, dim=1), dim=-1)
    (center_max, idx) = th.max(distance, dim=-1)
    center_mean_loss = th.mean(distance)
    center_max_loss = th.mean(center_max)
    center_loss = center_max_loss
    cross_matrix = cdist(person_center, person_center)
    cross_mask = th.ones_like(cross_matrix) - th.eye(batch_person).to('cuda')
    cross_vector = th.masked_select(cross_matrix, cross_mask.byte())
    # print('size', cross_vector.size())
    min_cross, idx = th.min(cross_vector.view(batch_person, -1), -1)       #其它类中心与目标最小距离
    cross_loss = th.mean(min_cross)

    return center_loss, cross_loss, center_loss





def CenterHardLoss2(fc, batch_person, num_file, fcs=128):
    fc = fc.view(batch_person, num_file, fcs)
    person_center = th.unsqueeze(th.unsqueeze(th.mean(input=fc, dim=1), dim=1), dim=3)
    fc = fc.view(1, num_file, fcs, batch_person)
    distance = th.norm(fc - person_center, dim=2)
    center_mask = th.unsqueeze(th.eye(batch_person), 1).expand_as(distance).to('cuda')
    cross_mask = center_mask * 1e6 + th.zeros_like(distance).to('cuda')
    center_matrix = th.masked_select(distance, center_mask.byte()).view(batch_person, -1)
    cross_matrix = cross_mask + distance
    center_loss_mean = th.mean(th.mean(center_matrix, -1))
    (center_max, idx) = th.max(center_matrix, -1)
    center_loss_max = th.mean(center_max)
    center_loss = 0.5 * (center_loss_mean + center_loss_max)
    (cross_loss, idx) = th.min(cross_matrix, dim=-1)
    (cross_loss, idx) = th.min(cross_loss, dim=-1)
    cross_loss = th.mean(cross_loss)
    return center_loss, cross_loss, center_loss / cross_loss



def CenterSemihardLoss(fc, batch_person, num_file, margin, fcs=128):
    fc = fc.view(batch_person, num_file, fcs)
    person_center = th.unsqueeze(th.unsqueeze(th.mean(input=fc, dim=1), dim=1), dim=3)
    fc = fc.view(1, num_file, fcs, batch_person)
    distance = th.norm(fc - person_center, dim=2)
    center_mask = th.unsqueeze(th.eye(batch_person), 1).expand_as(distance).to('cuda')
    cross_matrix_mask = center_mask * 1e6 + th.zeros_like(distance).to('cuda')
    center_matrix = th.masked_select(distance, center_mask.byte()).view(batch_person, -1)
    cross_matrix = cross_matrix_mask + distance
    low_boundary = th.transpose(th.unsqueeze(center_matrix, 0), 1, 2).expand_as(cross_matrix) - margin
    up_boundary = low_boundary + margin
    center_loss = th.mean(th.mean(center_matrix, -1))

    cross_loss_mask_up = th.where(cross_matrix < up_boundary, th.ones_like(distance), th.zeros_like(distance))
    cross_loss_mask_low = th.where(cross_matrix > low_boundary, th.ones_like(distance), th.zeros_like(distance))
    cross_loss_mask = cross_loss_mask_low * cross_loss_mask_up
    cross_loss_list = th.masked_select(distance, cross_loss_mask.byte())
    if len(cross_loss_list) > 0:
        cross_loss = th.mean(cross_loss_list)
    else:
        (cross_min, idx) = th.min(cross_matrix, -1)
        (cross_min, idx) = th.min(cross_min, -1)
        cross_loss = th.mean(cross_min)
    num_hards = len(cross_loss_list)
    loss_r = center_loss - cross_loss + margin
    loss = th.where(loss_r > 0, loss_r, th.zeros_like(loss_r))

    return center_loss, cross_loss, loss, num_hards / batch_person







def batch_hard(fc, pids):
    dists = cdist(fc, fc)
    batch_size = len(pids)
    same_identity_mask = (th.eq(th.unsqueeze(pids, dim=1), th.unsqueeze(pids, dim=0)))
    negative_mask = th.ones_like(same_identity_mask).to('cuda') - same_identity_mask  # 类间距离，越大越好
    positive_mask = same_identity_mask - th.eye(len(pids), dtype=th.uint8).to('cuda')  # 类内距离，越小越好
    negative_matrix = th.masked_select(dists, negative_mask.byte()).view(batch_size, -1)
    positive_matrix = th.masked_select(dists, positive_mask.byte()).view(batch_size, -1)
    (max_positive, idx) = th.max(positive_matrix, dim=-1); max_positive = th.unsqueeze(max_positive, dim=1)
    hards_mask = th.where(negative_matrix < max_positive.expand_as(negative_matrix),
                          th.ones_like(negative_matrix), th.zeros_like(negative_matrix))
    hards = th.masked_select(negative_matrix, hards_mask.byte())
    return len(hards)/batch_size











def CenterSemihardLoss2(fc, batch_person, num_file, fcs=128):
    fc = fc.view(batch_person, num_file, fcs)
    person_center = th.unsqueeze(th.unsqueeze(th.mean(input=fc, dim=1), dim=1), dim=3)
    fc = fc.view(1, num_file, fcs, batch_person)
    distance = th.norm(fc - person_center, dim=2)
    center_mask = th.unsqueeze(th.eye(batch_person), 1).expand_as(distance).to('cuda')
    cross_matrix_mask = center_mask * 1e6 + th.zeros_like(distance).to('cuda')
    center_matrix = th.masked_select(distance, center_mask.byte()).view(batch_person, -1)
    cross_matrix = cross_matrix_mask + distance

    center_mean_loss = th.mean(th.mean(center_matrix, -1))
    (max_center, idx) = th.max(center_matrix, -1)
    center_max_loss = th.mean(max_center)
    center_loss = 0.5 * (center_mean_loss + center_max_loss)
    cross_loss_mask = th.where(cross_matrix < max_center.expand_as(distance), th.ones_like(distance), th.zeros_like(distance))
    cross_loss_list = th.masked_select(cross_matrix, cross_loss_mask.byte())
    if len(cross_loss_list) > 0:
        cross_loss = th.mean(cross_loss_list)
    else:
        (cross_min, idx) = th.min(cross_matrix, -1)
        cross_loss = th.mean(cross_min)
    num_hards = len(cross_loss_list)
    return center_loss, cross_loss, center_loss / cross_loss, num_hards