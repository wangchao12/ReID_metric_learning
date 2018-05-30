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



def batch_hard(dists, pids, margin, k):
    same_identity_mask = (torch.eq(torch.unsqueeze(pids, dim=1),
                                   torch.unsqueeze(pids, dim=0)))
    negative_mask = torch.ones_like(same_identity_mask).to('cuda') - same_identity_mask       # 类间距离，越大越好
    positive_mask = same_identity_mask - torch.eye(len(pids), dtype=torch.uint8).to('cuda')   #类内距离，越小越好
    positive_top_value, idx = torch.topk(torch.masked_select(dists, positive_mask.byte()), k) #类内取最大的k项
    negative_top_value, idx = torch.topk(-torch.masked_select(dists, negative_mask.byte()), k)
    positive_dists = torch.mean(positive_top_value)
    negative_dists = torch.mean(-negative_top_value)

    return positive_dists, negative_dists, positive_dists - negative_dists + margin



def batch_easy(dists, pids, margin):

    same_identity_mask = (torch.eq(torch.unsqueeze(pids, dim=1),
                                  torch.unsqueeze(pids, dim=0)))
    negative_mask = torch.ones_like(same_identity_mask).to('cuda') - same_identity_mask     #类间距离，越大越好
    positive_mask = same_identity_mask - torch.eye(len(pids), dtype=torch.uint8).to('cuda') #类内距离，越小越好
    positive_dists = torch.mean(torch.masked_select(dists, positive_mask.byte()))           #类内均值，类间均值，平庸状况
    negative_dists = torch.mean(torch.masked_select(dists, negative_mask.byte()))
    return positive_dists, negative_dists, positive_dists - negative_dists + margin



def TripletHardLoss(fc, pids, margin,k):
    all_dists = cdist(fc, fc)
    pos, neg, loss = batch_hard(all_dists, pids, margin, k)
    return pos, neg, loss



def TripletEasyLoss(fc, pids, margin):
    all_dists = cdist(fc, fc)
    pos, neg, loss = batch_easy(all_dists, pids, margin)
    return pos, neg, loss