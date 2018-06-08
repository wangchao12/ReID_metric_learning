import cv2
import os
import torch
import numpy as np
import scipy.io as sio



def imgs_to_pt(path):
    data_list = []
    last_id = []
    person_list = []
    files = os.listdir(path)
    for i, file_i in enumerate(files):
        file_i_s = file_i.split('_', len(file_i))
        current_id = file_i_s[0]
        if current_id == last_id:
            img = cv2.imread(os.path.join(path, file_i))
            person_list.append(img)
        else:
            if len(person_list) > 0:
                data_list.append(person_list)
            person_list = []
            img = cv2.imread(os.path.join(path, file_i))
            person_list.append(img)
            file_i_s = file_i.split('_', len(file_i))
            last_id = file_i_s[0]
    final_list = []
    for i, persion_i in enumerate(data_list):
        person_list2 = []
        for file_i in persion_i:
            dict={'image': file_i, 'id': i}
            person_list2.append(dict)
        final_list.append(person_list2)
    return final_list

def img_to_test(path):
    file_list = []
    files = os.listdir(path)
    for idx, file_i in enumerate(files):
        print(idx)
        ids = file_i.split('_', len(file_i))
        id = ids[0]
        img = cv2.imread(os.path.join(path, file_i))
        dict_i = {'image': img, 'id': id}
        file_list.append(dict_i)
    return file_list



if __name__ == '__main__':
    file_path_train1 = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_train\\'
    file_path_train2 = 'E:\Person_ReID\DataSet\DukeMTMC-reID\DukeMTMC-reID\\train_128_64\\'
    file_path_test = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_test\\'
    person_list1 = imgs_to_pt(path=file_path_train1)
    person_list2 = imgs_to_pt(path=file_path_train2)
    person_list3 = imgs_to_pt(path=file_path_test)
    train_list = person_list1 + person_list2
    torch.save(train_list, '.\\traindata.pt')
    torch.save(person_list3, '.\\testdata.pt')




    # # print(np.shape(person_list))
    # query_path = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\query\\'
    # gallary_path = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_train\\'
    # # query_list = img_to_test(query_path)
    # train_list = img_to_test(gallary_path)
    # # torch.save(query_list, 'query.pt')
    # torch.save(train_list, 'train.pt')



