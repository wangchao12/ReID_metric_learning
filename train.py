from DataLoader import DataLoader
from models.mobilenet import *
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from SummaryWriter import SummaryWriter
from Loss2 import CenterEasyLoss4, batch_hard, TripletHardLoss
###parameters setting###
batch_person = 32
person_size = 8
epoches = 100000
margin = 0.1
scale = 0.5


trainloader = DataLoader(datafile='.\dataset\\traindata.pt', batch_person=batch_person, person_size=person_size)
testloader = DataLoader(datafile='.\dataset\\testdata.pt', batch_person=batch_person, person_size=person_size)
writer = SummaryWriter('.\log\log.mat')


model = MobileNetV2().to('cuda')
model.train()
model.load_state_dict(torch.load('.\checkpoint\\ReID_HardModel196.pt'))
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
        fc = model(torch.cuda.FloatTensor(batch_x))
        center_loss, cross_loss, loss1 = TripletHardLoss(fc, pids)
        loss = loss1
        loss.backward()
        optresnet.step()
        writer.write('trainHardLoss', float(loss))
        print('train epoch', i, 'iter', j, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss))
    sum_loss = 0
    ###############test stage################################
    for k in range(testloader.num_step):
        test_x, label = testloader.next_batch()
        fc = model.forward(torch.cuda.FloatTensor(test_x))
        center_loss, cross_loss, loss1 = TripletHardLoss(fc, pids)
        loss = loss1
        sum_loss = sum_loss + float(loss)
        writer.write('testHardLoss', float(loss))
        print('test epoch', i, 'iter', k, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss))
    print('min_test_loss', min_test_loss, 'test_loss', sum_loss / testloader.num_step)
    if sum_loss / testloader.num_step < min_test_loss:
        min_test_loss = sum_loss / testloader.num_step
        print('**************save model*******************')
        torch.save(model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
    writer.savetomat()
