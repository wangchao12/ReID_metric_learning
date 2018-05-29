from DataLoader import DataLoader
from models.Resnet import resnet18
import torch.optim as optim
from Loss import *

batch_person = 10
person_size = 5
epoches = 100




trainloader = DataLoader(datafile='traindata.pt', batch_person=batch_person, person_size=person_size)
model = resnet18()
optimizer = optim.Adadelta(model.parameters(), lr=0.001, rho=0.7)


for i in range(epoches):

    batch_x = trainloader.next_batch()
    model.forward(batch_x)
    loss = TripletLoss()


