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


def person_to_pt(path, label):
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
    for persion_i in data_list:
        person_list2 = []
        for file_i in persion_i:
            dict={'image': file_i, 'label': label}
            person_list2.append(dict)
        final_list.append(person_list2)
    attribute_list = []
    for person_i in final_list:
        for file_i in person_i:
            attribute_list.append(file_i)
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
    file_path_train = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_train_mask\\'
    file_path_train2 = 'E:\Person_ReID\DataSet\DukeMTMC-reID\DukeMTMC-reID\\train_128_64_mask\\'
    file_path_train3 = 'E:\Person_ReID\DataSet\cuhk03_release\labeled_mask\\'
    file_path_train4 = 'E:\Person_ReID\DataSet\DukeMTMC-reID\DukeMTMC-reID\\test_128_64_mask\\'
    file_path_test = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_test'


    train_list1 = imgs_to_pt(path=file_path_train)
    print('1')
    train_list2 = imgs_to_pt(path=file_path_train2)
    print('2')
    train_list3 = imgs_to_pt(path=file_path_train3)
    print('3')
    train_list4 = imgs_to_pt(path=file_path_train4)
    print('4')
    train_list = train_list1 + train_list2 + train_list3 + train_list4
    test_list = imgs_to_pt(path=file_path_test)
    print('5')
    torch.save(train_list, '.\\traindata.pt')
    print('6')
    torch.save(test_list, '.\\testdata.pt')





    # # print(np.shape(person_list))
    # query_path = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\query\\'
    # gallary_path = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_train\\'
    # # query_list = img_to_test(query_path)
    # train_list = img_to_test(gallary_path)
    # # torch.save(query_list, 'query.pt')
    # torch.save(train_list, 'train.pt')



