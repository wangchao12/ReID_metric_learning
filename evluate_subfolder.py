from models.mobilenet_multiway2 import MobileNetV2
import torch as th
import time, os, cv2
import numpy as np
import scipy.io as sio




model = MobileNetV2().to('cuda')
model.eval()
model.load_state_dict(th.load('.\checkpoint\\Model_mask174.pt'))
######################################################################
######################################################################
model_mask = MobileNetV2().to('cuda')
model_mask.eval()
model_mask.load_state_dict(th.load('.\checkpoint\\ReID_HardModel87.pt'))


def extract_fc(img):
    t1 = time.time()
    img = np.expand_dims(np.transpose(img, [2, 0, 1]), 0)
    output = model(th.cuda.FloatTensor(img))
    output_mask = model_mask(th.cuda.FloatTensor(output[-1]))
    cat_fc = th.cat((output[0], output_mask[0]), -1)
    fc = cat_fc / th.unsqueeze(th.norm(cat_fc, 2, -1), -1)
    t2 = time.time()
    print('time', t2 - t1)
    return fc


def extract_gallary(gallary_path):
    gallary_list = []
    persons = os.listdir(gallary_path)
    for person_i in persons:
        try:
            files = os.listdir(os.path.join(gallary_path, person_i))
            person_list = []
            for file_i in files:
                img_path = os.path.join(gallary_path, person_i, file_i)
                img = cv2.imread(img_path)
                fc = extract_fc(img)
                dict_i = {'image': img, 'feature': fc.detach().cpu().numpy(), 'id': person_i}
                person_list.append(dict_i)
            gallary_list.append(person_list)
        except:
            continue
    return gallary_list


def extract_query(query_path):
    query_list = []
    files = os.listdir(query_path)
    for file_i in files:
        img_path = os.path.join(query_path, file_i)
        img = cv2.imread(img_path)
        fc = extract_fc(img)
        dict_i = {'image': img, 'feature': fc.detach().cpu().numpy()}
        query_list.append(dict_i)
    return query_list


if __name__ =='__main__':
    gallary = 'E:\Person_ReID\DataSet\SmartVision_test_dataset\\xuchang\gallary'
    query = 'E:\Person_ReID\DataSet\SmartVision_test_dataset\\xuchang\query'
    gallary_list = extract_gallary(gallary)
    query_list = extract_query(query)
    sio.savemat('./evulate/query_xuchang_subway.mat', {'query_mask_xuchang': query_list})
    sio.savemat('./evulate/gallary_xuchang_subway.mat', {'gallary_mask_xuchang': gallary_list})


