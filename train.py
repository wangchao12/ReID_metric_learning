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
model.load_state_dict(torch.load('.\checkpoint\\ReID_HardModel47.pt'))
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
        global_emb, global_cls, sub21_emb, sub21_cls, sub22_emb, sub22_cls, \
        sub31_emb, sub31_cls, sub32_emb, sub32_cls, sub33_emb, sub33_cls, all_emb = model(torch.cuda.FloatTensor(batch_x))
        loss_global_cls = nn.CrossEntropyLoss()(global_cls, torch.cuda.LongTensor(label))
        loss_sub21_cls = nn.CrossEntropyLoss()(sub21_cls, torch.cuda.LongTensor(label))
        loss_sub22_cls = nn.CrossEntropyLoss()(sub22_cls, torch.cuda.LongTensor(label))
        loss_sub31_cls = nn.CrossEntropyLoss()(sub31_cls, torch.cuda.LongTensor(label))
        loss_sub32_cls = nn.CrossEntropyLoss()(sub32_cls, torch.cuda.LongTensor(label))
        loss_sub33_cls = nn.CrossEntropyLoss()(sub33_cls, torch.cuda.LongTensor(label))

        _, _, loss_global_emb, _ = CenterEasyLoss(global_emb, pids, batch_person, person_size, scale, margin)
        _, _, loss_sub21_emb, _ = CenterEasyLoss(sub21_emb, pids, batch_person, person_size, scale, margin)
        _, _, loss_sub22_emb, _ = CenterEasyLoss(sub22_emb, pids, batch_person, person_size, scale, margin)
        _, _, loss_sub31_emb, _ = CenterEasyLoss(sub31_emb, pids, batch_person, person_size, scale, margin)
        _, _, loss_sub32_emb, _ = CenterEasyLoss(sub32_emb, pids, batch_person, person_size, scale, margin)
        _, _, loss_sub33_emb, _ = CenterEasyLoss(sub33_emb, pids, batch_person, person_size, scale, margin)
        center_loss, cross_loss, loss_all_emb, n_hards = CenterEasyLoss(all_emb, pids, batch_person, person_size, scale, margin, fcs=768)

        loss = loss_global_emb + loss_sub21_emb + loss_sub22_emb + loss_sub31_emb + loss_sub32_emb + loss_sub33_emb + \
               loss_all_emb + loss_global_cls + loss_sub21_cls + loss_sub22_cls + loss_sub31_cls + loss_sub32_cls + loss_sub33_cls

        loss.backward()
        optresnet.step()
        writer.write('trainLoss', float(loss))
        writer.write('loss_all_emb', float(loss_all_emb))
        writer.write('trainhards', float(n_hards))
        print('train epoch', i, 'iter', j, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', n_hards)

    sum_loss = 0
    ###############test stage################################
    for k in range(testloader.num_step):
        test_x, label = testloader.next_batch()
        _, _, _, _, _, _, _, _, _, _, _, _, all_emb = model(torch.cuda.FloatTensor(test_x))
        center_loss, cross_loss, loss, n_hards = CenterEasyLoss(all_emb, pids, batch_person, person_size, scale, margin, fcs=768)
        writer.write('testLoss', float(loss))
        writer.write('testHards', float(n_hards))
        print('test epoch', i, 'iter', k, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', n_hards)
        sum_loss += float(loss)
    if sum_loss / testloader.num_step < min_test_loss:
        print('min_test_loss', min_test_loss, 'test_loss', sum_loss / testloader.num_step)
        min_test_loss = sum_loss / testloader.num_step
        print('**************save model*******************')
        torch.save(model.model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
    writer.savetomat()