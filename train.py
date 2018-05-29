from DataLoader import DataLoader
from models.Resnet import resnet18
import torch.optim as optim
import torch
import numpy as np
from Loss import TripletEasyLoss, TripletHardLoss

###parameters setting###

batch_person = 10
person_size = 5
epoches = 100




trainloader = DataLoader(datafile='traindata.pt', batch_person=batch_person, person_size=person_size)
model = resnet18().to('cuda')
optresnet = optim.Adadelta(model.parameters(), lr=0.001, rho=0.7)
pids_n = np.ones(batch_person * person_size)
for i in range(batch_person):
    pids_n[i: i+person_size] = i
pids = torch.from_numpy(pids_n).to('cuda')


for i in range(epoches):
    print(i)
    batch_x = trainloader.next_batch()
    fc = model.forward(torch.cuda.FloatTensor(batch_x))
    loss = TripletHardLoss(fc=fc, pids=pids, margin=5)
    loss.backward()
    optresnet.step()


