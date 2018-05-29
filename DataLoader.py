import torch
import numpy as np
import random
import math
import scipy.io as sio


class DataLoader(object):
    def __init__(self, datafile, batch_person, person_size):
        self.person_list = torch.load(datafile)
        self.bp = batch_person
        self.ps = person_size
        self.step = 1

    def shuffle_data(self):
        random.shuffle(self.person_list)
        self.step = 1

    def next_batch(self):
        batch_x = []
        if self.step > int(math.floor(len(self.person_list) / self.bp)):
            self.shuffle_data()
        start = (self.step - 1) * self.bp
        stop = self.step * self.bp
        persons = self.person_list[start:stop]
        for person_i in persons:
            idx = np.random.randint(low=1, high=len(person_i), size=self.ps)
            imgs = [np.transpose(a=person_i[i], axes=[2, 0, 1]) for i in idx]
            imgs = np.array(imgs)
            batch_x.append(imgs)
        batch_x = np.array(batch_x)
        batch_x = np.reshape(a=batch_x, newshape=(self.bp * self.ps, 3, 128, 64))
        self.step += 1
        return batch_x

if __name__ == '__main__':
    trainLoader = DataLoader('traindata.pt', 10, 5)
    batch_x = trainLoader.next_batch()
    dic = {'batch': batch_x}
    sio.savemat('batch.mat', dic)



