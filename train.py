from DataLoader import DataLoader
from models.mobilenet_mask import *
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from SummaryWriter import SummaryWriter
from Loss import CenterEasyLoss4, CenterEasyLoss5, MSE_loss
###parameters setting###
batch_person = 16
person_size = 8
epoches = 100000
margin = 0.1
scale = 0.5


trainloader = DataLoader(datafile='.\dataset\\traindata.pt', batch_person=batch_person, person_size=person_size)
testloader = DataLoader(datafile='.\dataset\\testdata.pt', batch_person=batch_person, person_size=person_size)
writer = SummaryWriter('.\log\log.mat')


model = MobileNetV2().to('cuda')
model2 = MobileNetV2().to('cuda')
model2.train()
model.train()
# model.load_state_dict(torch.load('.\checkpoint\\\ReID_HardModel258.pt'))
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
        fc, mask = model(torch.cuda.FloatTensor(batch_x))
        fc_, mask2 = model2(torch.cuda.FloatTensor(batch_x) * mask)
        center_loss, cross_loss, loss_g, n_hards = CenterEasyLoss4(fc, pids, batch_person, person_size, scale, margin, 128)
        center_loss, cross_loss, loss_m, n_hards = CenterEasyLoss4(fc_, pids, batch_person, person_size, scale, margin,
                                                                   128)
        loss_se = MSE_loss(fc, fc_)
        loss = loss_g + loss_m + loss_se
        loss.backward()
        optresnet.step()
        writer.write('trainHardLoss', float(loss))
        print('train epoch', i, 'iter', j, 'loss_g', float(loss_g), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', n_hards)
        print('train epoch', i, 'iter', j, 'loss_m', float(loss_se), 'center_loss_m')

    sum_loss = 0
    ###############test stage################################
    for k in range(testloader.num_step):
        test_x, label = testloader.next_batch()
        fc, mask = model(torch.cuda.FloatTensor(batch_x))
        fc_, mask2 = model(torch.cuda.FloatTensor(batch_x) * mask)
        center_loss, cross_loss, loss_g, n_hards = CenterEasyLoss4(fc, pids, batch_person, person_size, scale, margin)
        center_loss, cross_loss, loss_m, n_hards = CenterEasyLoss4(fc_, pids, batch_person, person_size, scale, margin,
                                                                   128)
        loss_se = MSE_loss(fc, fc_)
        loss = loss_g + loss_m + loss_se
        sum_loss = sum_loss + float(loss)
        writer.write('testHardLoss', float(loss))
        writer.write('testHards', float(n_hards))
        print('test epoch', i, 'iter', k, 'loss_g', float(loss_g), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', n_hards)
        print('test epoch', i, 'iter', k, 'loss_m', float(loss_se), 'center_loss_m')

    print('min_test_loss', min_test_loss, 'test_loss', sum_loss / testloader.num_step)
    if sum_loss / testloader.num_step < min_test_loss:
        min_test_loss = sum_loss / testloader.num_step
        print('**************save model*******************')
        torch.save(model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
    writer.savetomat()