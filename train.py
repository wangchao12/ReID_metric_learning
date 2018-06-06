from DataLoader import DataLoader
from models.mobilenet import *
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from SummaryWriter import SummaryWriter
from Loss2 import CenterEasyLoss3, CenterSemihardLoss, batch_hard
from Loss.TripletLoss import TripletLoss
from Loss.loss import global_loss
###parameters setting###
batch_person = 32
person_size = 8
epoches = 100000
margin = 0.5


trainloader = DataLoader(datafile='.\dataset\\traindata.pt', batch_person=batch_person, person_size=person_size)
testloader = DataLoader(datafile='.\dataset\\testdata.pt', batch_person=batch_person, person_size=person_size)
writer = SummaryWriter('.\log\log.mat')


model = MobileNetV2(n_classes=len(trainloader.person_list)).to('cuda')
# model.load_state_dict(torch.load('.\checkpoint\ReID_HardModel63.pt'))
optresnet = optim.Adadelta(model.parameters(), lr=1e-3)
pids_n = []

for i in range(batch_person):
    pids_n.append(i * np.ones(person_size))
pids_n = np.reshape(a=np.array(pids_n), newshape=-1)
pids = torch.from_numpy(pids_n).to('cuda')
min_test_loss = 1e6
for i in range(epoches):
    iter = 0
    ###################################train stage###############################################
    for j in range(trainloader.num_step):
        iter += 1
        batch_x, label = trainloader.next_batch()
        fc, cls = model.forward(torch.cuda.FloatTensor(batch_x))
        center_loss, cross_loss, loss1 = CenterEasyLoss3(fc, batch_person, person_size, 128)
        loss2 = nn.CrossEntropyLoss().forward(cls, torch.cuda.LongTensor(label))
        num_hards = batch_hard(fc, pids)
        loss = loss1 + loss2
        loss.backward()
        optresnet.step()
        writer.write('trainHardLoss', float(loss))
        writer.write('traincenterLoss', float(loss1))
        writer.write('trainclsLoss', float(loss2))
        writer.write('trainhards', float(num_hards))
        print('train epoch', i, 'iter', j, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', num_hards)
    sum_loss = 0
    ###############test stage################################
    for k in range(testloader.num_step):
        test_x, label = testloader.next_batch()
        fc, cls = model.forward(torch.cuda.FloatTensor(test_x))
        center_loss, cross_loss, loss1 = CenterEasyLoss3(fc, batch_person, person_size, 128)
        loss2 = nn.CrossEntropyLoss().forward(cls, torch.cuda.LongTensor(label))
        loss = loss1 + loss2
        num_hards = batch_hard(fc, pids)
        sum_loss = sum_loss + float(loss)
        writer.write('testHardLoss', float(loss))
        writer.write('testcenterLoss', float(loss1))
        writer.write('testclsLoss', float(loss2))
        writer.write('testhards', float(num_hards))
        print('test epoch', i, 'iter', k, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', num_hards)
    print('min_test_loss', min_test_loss, 'test_loss', sum_loss / testloader.num_step)
    if sum_loss / testloader.num_step < min_test_loss:
        min_test_loss = sum_loss / testloader.num_step
        print('**************save model*******************')
        torch.save(model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
    writer.savetomat()
