from DataLoader import DataLoader
from models.mobilenet import *
import torch.optim as optim
import torch
import numpy as np
from Loss import TripletHardLoss
from SummaryWriter import SummaryWriter


###parameters setting###
batch_person = 16
person_size = 8
epoches = 1000
margin = 0.5


trainloader = DataLoader(datafile='.\dataset\\traindata.pt', batch_person=batch_person, person_size=person_size)
testloader = DataLoader(datafile='.\dataset\\testdata.pt', batch_person=batch_person, person_size=person_size)
writer = SummaryWriter('.\log\log.mat')


model = MobileNetV2().to('cuda')
model.load_state_dict(torch.load('.\checkpoint\ReID_HardModel57.pt'))
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
        loss, num_hards = TripletHardLoss(fc=fc, pids=pids, margin=margin)
        loss.backward()
        optresnet.step()
        writer.write('trainHardLoss', float(loss))
        print('train epoch', i, 'iter', j, 'loss', float(loss), 'num_hards', float(num_hards))
    sum_loss = 0
    for k in range(testloader.num_step):
        test_x = testloader.next_batch()
        fc = model.forward(torch.cuda.FloatTensor(test_x))
        loss, num_hards = TripletHardLoss(fc=fc, pids=pids, margin=margin)
        sum_loss = sum_loss + float(loss)
        writer.write('testHardLoss', float(loss))
        print('test epoch', i, 'iter', k, 'loss', float(loss), 'num_hards', float(num_hards))
    print('min_test_loss', min_test_loss)
    if sum_loss / testloader.num_step < min_test_loss:
        min_test_loss = sum_loss / testloader.num_step
        print('**************save model*******************')
        torch.save(model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
        writer.savetomat()
