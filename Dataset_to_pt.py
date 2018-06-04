import cv2
import os
import torch
import numpy as np



def imgs_to_pt(path):
    data_list = []
    person_list = []
    last_id = ' '
    files = os.listdir(path)
    for file_i in files:
        current_id = file_i[0:4]
        if current_id == last_id:
            img = cv2.imread(os.path.join(path, file_i))
            person_list.append(img)
        else:
            if len(person_list) > 0:
                data_list.append(person_list)
            person_list = []
            img = cv2.imread(os.path.join(path, file_i))
            person_list.append(img)
            last_id = file_i[0:4]
    return data_list

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
    file_path_train = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_train\\'
    file_path_test = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_test\\'
    person_list = imgs_to_pt(path=file_path_train)
    np.savez('traindata.npz', person_list)
    person_list2 = imgs_to_pt(path=file_path_test)
    np.savez('testdata.npz', person_list2)



    # print(np.shape(person_list))
    # query_path = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\query\\'
    # gallary_path = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_test\\'
    # query_list = img_to_test(query_path)
    # gallary_list = img_to_test(gallary_path)
    # torch.save(query_list, 'query.pt')
    # torch.save(gallary_list, 'gallary.pt')



