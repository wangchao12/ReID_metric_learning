import cv2
import os
import torch



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


if __name__ == '__main__':
    file_path_train = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_train\\'
    file_path_test = 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_test\\'
    person_list = imgs_to_pt(path=file_path_train)
    torch.save(person_list, 'traindata.pt')
    person_list2 = imgs_to_pt(path=file_path_test)
    torch.save(person_list2[4:-1], 'testdata.pt')
    # print(np.shape(person_list))


