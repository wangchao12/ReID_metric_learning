import torch
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



def TripletHardLoss(dists, pids):

    same_identity_mask = torch.eq(torch.unsqueeze(pids, axis=1),
                                  torch.unsqueeze(pids, axis=0))
    negative_mask = torch.ones_like(same_identity_mask) - same_identity_mask    #类间距离，越大越好
    positive_mask = same_identity_mask - torch.eye(len(pids))                   #类内距离，越小越好
    positive_dists = torch.mean(torch.min(dists * positive_mask, axis=1))       #类内最小，类间最大，非常极端
    negative_dists = torch.mean(torch.max(dists * negative_mask, axis=1))
    return positive_dists / negative_dists


def TripletLoss(dists, pids):

    same_identity_mask = torch.eq(torch.unsqueeze(pids, axis=1),
                                  torch.unsqueeze(pids, axis=0))
    negative_mask = torch.ones_like(same_identity_mask) - same_identity_mask    #类间距离，越大越好
    positive_mask = same_identity_mask - torch.eye(len(pids))                   #类内距离，越小越好
    positive_dists = torch.mean(dists * positive_mask, axis=1)                  #类内均值，类间均值，平庸状况
    negative_dists = torch.mean(dists * negative_mask, axis=1)
    return positive_dists / negative_dists