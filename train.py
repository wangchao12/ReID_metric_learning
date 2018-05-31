from DataLoader import DataLoader
from models.Resnet import *
import torch.optim as optim
import torch
import numpy as np
from Loss import TripletEasyLoss, TripletHardLoss
from SummaryWriter import SummaryWriter


###parameters setting###
batch_person = 16
person_size = 8
epoches = 1000
margin = 0


trainloader = DataLoader(datafile='.\dataset\\traindata.pt', batch_person=batch_person, person_size=person_size)
testloader = DataLoader(datafile='.\dataset\\testdata.pt', batch_person=batch_person, person_size=person_size)
writer = SummaryWriter('.\log\log.mat')


model = resnet34().to('cuda')
# model.load_state_dict(torch.load('.\checkpoint\ReID_HardModel380.pt'))
optresnet = optim.Adam(model.parameters(), lr=1e-5)
pids_n = []

for i in range(batch_person):
    pids_n.append(i * np.ones(person_size))
pids_n = np.reshape(a=np.array(pids_n), newshape=-1)
pids = torch.from_numpy(pids_n).to('cuda')
min_test_loss = 1e6
for i in range(epoches):
    iter = 0
    for j in range(trainloader.num_step):
        iter += 1
        batch_x = trainloader.next_batch()
        fc = model.forward(torch.cuda.FloatTensor(batch_x))
        output = TripletHardLoss(fc=fc, pids=pids, margin=margin)
        output['loss'].backward()
        optresnet.step()
        writer.write('trainHardLoss', float(output['loss']))
        print('train epoch', i, 'iter', j, 'loss', float(output['loss']),
              'pos', float(output['pos_dist']), 'neg', float(output['neg_dist']),
              'hard_pos', float(output['hard_pos']), 'hard_neg', float(output['hard_neg']))
    sum_loss = 0
    for k in range(testloader.num_step):
        test_x = testloader.next_batch()
        fc = model.forward(torch.cuda.FloatTensor(test_x))
        output = TripletHardLoss(fc=fc, pids=pids, margin=margin)
        sum_loss = sum_loss + float(output['loss'])
        writer.write('testHardLoss', float(output['loss']))
        print('test epoch', i, 'iter', k, 'loss', float(output['loss']),
              'pos', float(output['pos_dist']), 'neg', float(output['neg_dist']),
              'hard_pos', float(output['hard_pos']), 'hard_neg', float(output['hard_neg']))
    print('min_test_loss', min_test_loss)
    if sum_loss / testloader.num_step < min_test_loss:
        min_test_loss = sum_loss / testloader.num_step
        print('**************save model*******************')
        torch.save(model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
        writer.savetomat()

