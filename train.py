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


trainloader = DataLoader(datafile='traindata.pt', batch_person=batch_person, person_size=person_size)
writer = SummaryWriter('log.mat')


model = resnet18().to('cuda')
optresnet = optim.Adam(model.parameters(), lr=1e-5)
pids_n = []
for i in range(batch_person):
    pids_n.append(i * np.ones(person_size))
pids_n = np.reshape(a=np.array(pids_n), newshape=-1)
pids = torch.from_numpy(pids_n).to('cuda')

for i in range(epoches):
    iter = 0
    for j in range(trainloader.num_step):
        iter += 1
        batch_x = trainloader.next_batch()
        fc = model.forward(torch.cuda.FloatTensor(batch_x))
        pos, neg, loss = TripletEasyLoss(fc=fc, pids=pids, margin=0.5)
        total_loss = loss
        total_loss.backward()
        optresnet.step()
        writer.write('loss', float(loss))
        print('epoch', i, 'iter', j, 'total_loss', float(total_loss),
              'loss', float(loss), 'pos', float(pos), 'neg', float(neg))
    writer.savetomat()

