from DataLoader import DataLoader
from models.mobilenet_multiway import *
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
margin = 0.2
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
model.load_state_dict(torch.load('.\checkpoint\\ReID_HardModel87.pt'))
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
        global_emb, global_cls, sub1_emb, sub1_cls, sub2_emb, sub2_cls, all_emb = model(torch.cuda.FloatTensor(batch_x))
        loss_global_cls = nn.CrossEntropyLoss()(global_cls, torch.cuda.LongTensor(label))
        loss_subl_cls = nn.CrossEntropyLoss()(sub1_cls, torch.cuda.LongTensor(label))
        loss_sub2_cls = nn.CrossEntropyLoss()(sub2_cls, torch.cuda.LongTensor(label))
        _, _, loss_tri_global, _ = CenterEasyLoss5(global_emb, pids, batch_person, person_size, scale, margin)
        _, _, loss_tri_sub1, _ = CenterEasyLoss5(sub1_emb, pids, batch_person, person_size, scale, margin)
        _, _, loss_tri_sub2, _ = CenterEasyLoss5(sub2_emb, pids, batch_person, person_size, scale, margin)
        center_loss, cross_loss, loss_tri_all, n_hards = CenterEasyLoss5(all_emb, pids, batch_person, person_size, scale, margin, fcs=384)
        loss = loss_global_cls + loss_subl_cls + loss_sub2_cls + loss_tri_global + loss_tri_sub1 + loss_tri_sub2 + loss_tri_all
        loss.backward()
        optresnet.step()
        writer.write('trainLoss', float(loss))
        writer.write('loss_tri_all', float(loss_tri_all))
        writer.write('loss_tri_global', float(loss_tri_global))
        writer.write('loss_global_cls', float(loss_global_cls))
        writer.write('loss_subl_cls', float(loss_subl_cls))
        writer.write('loss_sub2_cls', float(loss_sub2_cls))
        writer.write('loss_tri_sub1', float(loss_tri_sub1))
        writer.write('loss_tri_sub2', float(loss_tri_sub2))
        writer.write('trainhards', float(n_hards))
        print('train epoch', i, 'iter', j, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', n_hards)

    sum_loss = 0
    ###############test stage################################
    for k in range(testloader.num_step):
        test_x, label = testloader.next_batch()
        _, _, _, _, _, _, all_emb = model(torch.cuda.FloatTensor(test_x))
        center_loss, cross_loss, loss, n_hards = CenterEasyLoss5(all_emb, pids, batch_person, person_size, scale, margin, fcs=384)
        writer.write('testLoss', float(loss))
        writer.write('testHards', float(n_hards))
        print('test epoch', i, 'iter', k, 'loss', float(loss), 'center_loss',
              float(center_loss), 'cross_loss', float(cross_loss), 'n_hards', n_hards)
        sum_loss += float(loss)
    if sum_loss / testloader.num_step < min_test_loss:
        min_test_loss = sum_loss / testloader.num_step
        print('min_test_loss', min_test_loss, 'test_loss', sum_loss / testloader.num_step)
        print('**************save model*******************')
        torch.save(model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
    writer.savetomat()