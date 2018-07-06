from models.mobilenet_multiway2 import MobileNetV2
import torch as th
import time
import numpy as np
import scipy.io as sio
from Dataset_to_pt import img_to_test



model = MobileNetV2().to('cuda')
model.eval()
model.load_state_dict(th.load('.\checkpoint\\\ReID_HardModel12.pt'))


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
            output = model(th.cuda.FloatTensor(img2))
            t2 = time.time()
            dict_i = {'img': img, 'fc': output[0].cpu().detach().numpy(), 'id': id}
            query_list.append(dict_i)
            print(idx, 'time:', t2 - t1)
        except:
            continue
    return {name: query_list}



def extract_fc_acc(query, name, batch_size):
    query_list = [];query_list_m = [];query_list_c = []
    num_steps = np.floor(len(query) / batch_size)
    for i in range(num_steps):
        start = i * batch_size; stop = (i + 1) * batch_size - 1
        query_list = query[start:stop]
        # for img in


    for idx, query_i in enumerate(query):
        try:
            img = query_i['image']
            id = query_i['id']
            img2 = np.expand_dims(np.transpose(img, [2, 0, 1]), axis=0)
            t1 = time.time()
            mask_fc, fc, cat_fc = model(th.cuda.FloatTensor(img2))
            t2 = time.time()
            dict_i = {'img': img, 'fc': fc.cpu().detach().numpy(), 'id': id}
            dict_i_m = {'img': img, 'fc': mask_fc.cpu().detach().numpy(), 'id': id}
            dict_i_cat = {'img': img, 'fc': cat_fc.cpu().detach().numpy(), 'id': id}
            query_list.append(dict_i)
            query_list_m.append(dict_i_m)
            query_list_c.append(dict_i_cat)
            print(idx, 'time:', t2 - t1)
        except:
            continue
    return {name: query_list}, {name: query_list_m}, {name: query_list_c}

if __name__ =='__main__':
    gallary_path = ['E:\Person_ReID\DataSet\SmartVision_test_dataset\subway\gallary_128_64',
                    'E:\Person_ReID\DataSet\SmartVision_test_dataset\detected_ped_images\gallary',
                    'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_test']

    query_path = ['E:\Person_ReID\DataSet\SmartVision_test_dataset\subway\query_128_64',
                  'E:\Person_ReID\DataSet\SmartVision_test_dataset\detected_ped_images\query',
                  'E:\Person_ReID\DataSet\Market-1501-v15.09.15\query']
    query_list = img_to_test(query_path[2])
    gallary_list = img_to_test(gallary_path[2])

    print('************Beganing test***************')
    gallary_fc = extract_fc(gallary_list, 'gallary')
    query_fc = extract_fc(query_list, 'query')
    sio.savemat('./evulate/query_1501.mat', query_fc)
    sio.savemat('./evulate/gallary_1501.mat', gallary_fc)



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

