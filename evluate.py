from models.mobilenet import MobileNetV2
import torch
import numpy as np
import time
import scipy.io as sio


model = MobileNetV2().to('cuda')
model.load_state_dict(torch.load('.\checkpoint\ReID_HardModel283.pt'))


def extract_fc(query, name):
    query_list = []
    for idx, query_i in enumerate(query):
        try:
            img = query_i['image']
            id = query_i['id']
            img2 = np.expand_dims(np.transpose(img, [2, 0, 1]), axis=0)
            t1 = time.time()
            fc = model.forward(torch.cuda.FloatTensor(img2))
            t2 = time.time()
            dict_i = {'img': img, 'fc': fc.cpu().detach().numpy(), 'id': id}
            query_list.append(dict_i)
            print(idx, 'time:', t2 - t1)
        except:
            continue
    return {name: query_list}


if __name__ =='__main__':
    query = torch.load('.\evulate\query.pt')
    gallary = torch.load('.\evulate\gallary.pt')
    print('*********************************')
    gallary_fc = extract_fc(gallary, 'gallary')
    query_fc = extract_fc(query, 'query')
    sio.savemat('./evulate/query_fc.mat', query_fc)
    sio.savemat('./evulate/gallary_fc.mat', gallary_fc)

