from DataLoader import DataLoader
from models.mobilenet_multiway2 import *
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from SummaryWriter import SummaryWriter
from Loss import CenterEasyLoss5 as CenterEasyLoss

###parameters setting###
batch_person = 16
person_size = 16
epoches = 100000
margin = 0.1
scale = 0.5
# 'E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_train\\',
trainList = ['E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_train\\',
             'E:\Person_ReID\DataSet\DukeMTMC-reID\DukeMTMC-reID\\train_128_64\\',
             'E:\Person_ReID\DataSet\cuhk03_release\labeled\\',
             'E:\Person_ReID\DataSet\DukeMTMC-reID\DukeMTMC-reID\\test_128_64\\']
testList = ['E:\Person_ReID\DataSet\Market-1501-v15.09.15\\bounding_box_test']

trainloader = DataLoader(datafile=trainList, batch_person=batch_person, person_size=person_size)
testloader = DataLoader(datafile=testList, batch_person=batch_person, person_size=person_size)
writer = SummaryWriter('.\log\log.mat')


model_base = MobileNetV2().to('cuda')
model_base.train()
model = ModelContainer(model_base).to('cuda')
model.train()
model.load_state_dict(torch.load('.\checkpoint\\ReID_HardModel67.pt'))
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
        output = model(torch.cuda.FloatTensor(batch_x))
        loss_cls = [nn.CrossEntropyLoss()(i, torch.cuda.LongTensor(label)) for i in output[4:7]]
        loss_tri = [CenterEasyLoss(i, pids, batch_person, person_size, scale, margin) for i in output[0:4]]
        loss = sum(loss_cls) + sum([loss_tri_[2] for loss_tri_ in loss_tri])
        loss.backward()
        optresnet.step()
        writer.write('trainTripletLoss', float(loss_tri[0][2]))
        writer.write('trainLoss', float(loss))
        writer.write('trainhards', float(loss_tri[0][3]))
        print('train epoch', i, 'iter', j, 'loss', float(loss), 'center_loss',
              float(loss_tri[0][0]), 'cross_loss', float(loss_tri[0][1]), 'n_hards', loss_tri[0][3])

    sum_loss = 0
    ###############test stage################################
    for k in range(testloader.num_step):
        test_x, label = testloader.next_batch()
        output = model(torch.cuda.FloatTensor(test_x))
        center_loss, cross_loss, loss, n_hards = CenterEasyLoss(output[0], pids, batch_person, person_size, scale, margin)
        writer.write('testLoss', float(loss))
        writer.write('testHards', float(n_hards))
        print('test epoch', i, 'iter', k, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', n_hards)
        sum_loss += float(loss)
    if sum_loss / testloader.num_step < min_test_loss:
        print('min_test_loss', min_test_loss, 'test_loss', sum_loss / testloader.num_step)
        min_test_loss = sum_loss / testloader.num_step
        print('**************save model*******************')
        torch.save(model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
    writer.savetomat()