import torch
import numpy as np
import random
import math
import scipy.io as sio


class DataLoader(object):
    def __init__(self, datafile, batch_size):
        self.person_list = torch.load(datafile)
        self.batch_size = batch_size

        self.step = 1
        self.num_step = int(np.ceil(len(self.person_list) / batch_size))

    def shuffle_data(self):
        random.shuffle(self.person_list)
        self.step = 1

    def next_batch(self):
        if self.step > int(math.floor(len(self.person_list) / self.batch_size)):
            self.shuffle_data()
        start = (self.step - 1) * self.batch_size
        stop = self.step * self.batch_size
        files = self.person_list[start:stop]
        imgs = [np.transpose(a=file_i['image'], axes=[2, 0, 1]) for file_i in files]
        labels = [file_i['attribute'] for file_i in files]
        batch_x = np.array(imgs)
        label = np.array(labels)
        batch_x = np.reshape(a=batch_x, newshape=(self.batch_size, 3, 128, 64))
        label = np.reshape(a=label, newshape=(self.batch_size, -1))
        self.step += 1
        return batch_x, label


if __name__ == '__main__':
    trainLoader = DataLoader('.\dataset\\traindata.pt', 10, 5)
    batch_x, label = trainLoader.next_batch()
    dic = {'batch': batch_x}
    sio.savemat('batch.mat', dic)



