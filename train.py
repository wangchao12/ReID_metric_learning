from DataLoader import DataLoader
from models.mobilenet_cat import *
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from SummaryWriter import SummaryWriter
from Loss import CenterEasyLoss4, CenterEasyLoss5

###parameters setting###
batch_person = 16
person_size = 16
epoches = 100000
margin = 0.1
scale = 0.5

trainList = ['E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_train\\',
             'E:\Person_ReID\DataSet\DukeMTMC-reID\DukeMTMC-reID\\train_128_64\\',
             'E:\Person_ReID\DataSet\cuhk03_release\labeled\\',
             'E:\Person_ReID\DataSet\DukeMTMC-reID\DukeMTMC-reID\\test_128_64\\']
testList = ['E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_test']

trainloader = DataLoader(datafile=trainList, batch_person=batch_person, person_size=person_size)
testloader = DataLoader(datafile=testList, batch_person=batch_person, person_size=person_size)
writer = SummaryWriter('.\log\log.mat')


model = MobileNetV2().to('cuda')
model.train()
# model_mask.load_state_dict(torch.load('.\checkpoint\\ReID_HardModel10.pt'))
optresnet = optim.Adadelta(model.parameters(), lr=1e-3)
pids_n = []

for i in range(batch_person):
    pids_n.append(i * np.ones(person_size, dtype=np.int32))
pids_n = np.reshape(a=np.array(pids_n), newshape=-1)
pids = torch.from_numpy(pids_n).to('cuda')
min_test_loss = 1e6
for i in range(epoches):
    iter = 0
    ###################################train stage###############################################
    for j in range(trainloader.num_step):
        iter += 1
        batch_x, label = trainloader.next_batch()
        fc, cls = model(torch.cuda.FloatTensor(batch_x))
        loss_cls = nn.CrossEntropyLoss()(cls, torch.cuda.LongTensor(label))
        center_loss, cross_loss, loss_tri, n_hards = CenterEasyLoss4(fc, pids, batch_person, person_size, scale, margin)
        loss = loss_cls + loss_tri
        loss.backward()
        optresnet.step()
        writer.write('trainLossCls', float(loss_cls))
        writer.write('trainLossTri', float(loss_tri))
        writer.write('trainhards', float(n_hards))
        print('train epoch', i, 'iter', j, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', n_hards)

    sum_loss = 0
    ###############test stage################################
    for k in range(testloader.num_step):
        test_x, label = testloader.next_batch()
        fc, mask = model(torch.cuda.FloatTensor(test_x))
        center_loss, cross_loss, loss, n_hards = CenterEasyLoss4(fc, pids, batch_person, person_size, scale, margin)
        writer.write('testLoss', float(loss))
        writer.write('testHards', float(n_hards))
        print('test epoch', i, 'iter', k, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', n_hards)
        sum_loss += float(loss)
    print('min_test_loss', min_test_loss, 'test_loss', sum_loss / testloader.num_step)
    if sum_loss / testloader.num_step < min_test_loss:
        min_test_loss = sum_loss / testloader.num_step
        print('**************save model*******************')
        torch.save(model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
    writer.savetomat()