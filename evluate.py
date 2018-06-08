from models.mobilenet import MobileNetV2
import torch as th
import numpy as np
import time
import numpy as np
import scipy.io as sio



model = MobileNetV2().to('cuda')
model.eval()
model.load_state_dict(th.load('E:\Person_ReID\ReID_metric_learning\checkpoint\\\ReID_HardModel16.pt'))


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


def extract_fc(query, name):
    query_list = []
    for idx, query_i in enumerate(query):
        try:
            img = query_i['image']
            id = query_i['id']
            img2 = np.expand_dims(np.transpose(img, [2, 0, 1]), axis=0)
            t1 = time.time()
            fc = model(th.cuda.FloatTensor(img2))
            t2 = time.time()
            dict_i = {'img': img, 'fc': fc.cpu().detach().numpy(), 'id': id}
            query_list.append(dict_i)
            print(idx, 'time:', t2 - t1)
        except:
            continue
    return {name: query_list}


if __name__ =='__main__':
    query = th.load('.\evulate\query.pt')
    gallary = th.load('.\evulate\\gallary.pt')
    print('*********************************')
    gallary_fc = extract_fc(gallary, 'gallary')
    query_fc = extract_fc(query, 'query')
    sio.savemat('./evulate/query_fc.mat', query_fc)
    sio.savemat('./evulate/gallary_fc.mat', gallary_fc)


#     model = MobileNetV2().to('cuda')
#     model.load_state_dict(th.load('E:\Person_ReID\ReID_metric_learning\checkpoint\\ReID_HardModel43.pt'))
#     model.eval()
#     query = th.load('.\evulate\query.pt')
#     train = th.load('.\dataset\\traindata.pt')
#     fcs = th.zeros([30, 128]).to('cuda')
#     batch_x = th.zeros([30, 3, 128, 64]).to('cuda')
#     for i in range(30):
#         img = train[i][0]['image']
#         img2 = np.expand_dims(np.transpose(img, [2, 0, 1]), axis=0)
#         batch_x[i, :, :, :] = th.cuda.FloatTensor(img2)
#         fc = model(th.cuda.FloatTensor(img2))
#         fcs[i, :] = fc
#     fc_batch = model(batch_x)
#     fcs_np = fcs.detach().cpu().numpy()
#     fc_batch_np = fc_batch.detach().cpu().numpy()
#     distance = cdist(fcs, fcs)
#     distance_batc
#
#     h = cdist(fc_batch, fc_batch)
#     distance_np = distance.detach().cpu().numpy()
#     distance_batch_np = distance_batch.detach().cpu().numpy()
#     print()

