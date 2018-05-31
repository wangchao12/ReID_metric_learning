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



def batch_hard(dists, pids, margin):
    batch_size = len(pids)
    same_identity_mask = (torch.eq(torch.unsqueeze(pids, dim=1),
                                   torch.unsqueeze(pids, dim=0)))
    negative_mask = torch.ones_like(same_identity_mask).to('cuda') - same_identity_mask       # 类间距离，越大越好
    positive_mask = same_identity_mask - torch.eye(len(pids), dtype=torch.uint8).to('cuda')   #类内距离，越小越好
    negative_matrix = torch.masked_select(dists, negative_mask.byte()).view(batch_size, -1)
    positive_matrix = torch.masked_select(dists, positive_mask.byte()).view(batch_size, -1)
    positive_max, positive_idx = torch.max(positive_matrix, dim=1)
    negative_min, negative_idx = torch.min(negative_matrix, dim=1)
    positive_max = torch.unsqueeze(positive_max, 1).expand_as(negative_matrix)
    negative_min = torch.unsqueeze(negative_min, 1).expand_as(positive_matrix)
    negative_hard_mask = torch.where(negative_matrix < positive_max + margin,
                         torch.ones_like(negative_matrix), torch.zeros_like(negative_matrix))
    positive_hard_mask = torch.where(positive_matrix > negative_min,
                         torch.ones_like(positive_matrix), torch.zeros_like(positive_matrix))
    positive_dists = torch.mean(torch.masked_select(positive_matrix, positive_hard_mask.byte()))
    negative_dists = torch.mean(torch.masked_select(negative_matrix, negative_hard_mask.byte()))
    n_hard_positive = len(torch.masked_select(positive_matrix, positive_hard_mask.byte()))
    n_hard_negaitive = len(torch.masked_select(negative_matrix, negative_hard_mask.byte()))

    output = {'pos_dist': positive_dists, 'neg_dist': negative_dists, 'hard_pos': n_hard_positive, 'hard_neg': n_hard_negaitive,
              'loss': positive_dists - negative_dists + 1}

    return output



def batch_easy(dists, pids, margin):

    same_identity_mask = (torch.eq(torch.unsqueeze(pids, dim=1),
                                  torch.unsqueeze(pids, dim=0)))
    negative_mask = torch.ones_like(same_identity_mask).to('cuda') - same_identity_mask     #类间距离，越大越好
    positive_mask = same_identity_mask - torch.eye(len(pids), dtype=torch.uint8).to('cuda') #类内距离，越小越好
    positive_dists = torch.mean(torch.masked_select(dists, positive_mask.byte()))           #类内均值，类间均值，平庸状况
    negative_dists = torch.mean(torch.masked_select(dists, negative_mask.byte()))
    return positive_dists, negative_dists, positive_dists - negative_dists + margin



def TripletHardLoss(fc, pids, margin):
    all_dists = cdist(fc, fc)
    output = batch_hard(all_dists, pids, margin)
    return output



def TripletEasyLoss(fc, pids, margin):
    all_dists = cdist(fc, fc)
    pos, neg, loss = batch_easy(all_dists, pids, margin)
    return pos, neg, loss