from DataLoader import DataLoader
from models.mobilenet import *
import torch.optim as optim
import torch
from SummaryWriter import SummaryWriter
from Loss import attribute_loss
###parameters setting###
batch_person = 32
person_size = 2
epoches = 100000
alpha = 15


trainloader = DataLoader(datafile='.\dataset\\traindata.pt', batch_person=batch_person, person_size=person_size)
testloader = DataLoader(datafile='.\dataset\\testdata.pt', batch_person=batch_person, person_size=person_size)
writer = SummaryWriter('.\log\log.mat')


model = MobileNetV2().to('cuda')
model.train()
# model.load_state_dict(torch.load('.\checkpoint\pre_train\\ReID_HardModel235.pt'))



optresnet = optim.Adadelta(model.parameters(), lr=1e-3)
min_test_loss = 1e6
for i in range(epoches):
    iter = 0
    ###################################train stage###############################################
    for j in range(trainloader.num_step):
        iter += 1
        batch_x, label = trainloader.next_batch()
        fc = model(torch.cuda.FloatTensor(batch_x))
        loss, n_err = attribute_loss(fc, torch.cuda.FloatTensor(label), alpha)
        loss.backward()
        optresnet.step()
        writer.write('trainLoss', float(loss))
        writer.write('trainHards', float(n_err))
        print('train epoch', i, 'iter', j, 'loss', float(loss), 'n_err', n_err)
    sum_loss = 0
    ###############test stage################################
    for k in range(testloader.num_step):
        test_x, label = testloader.next_batch()
        fc = model.forward(torch.cuda.FloatTensor(test_x))
        loss, n_err = attribute_loss(fc, torch.cuda.FloatTensor(label), alpha)
        sum_loss = sum_loss + float(loss)
        writer.write('testLoss', float(loss))
        writer.write('testHards', float(n_err))
        print('test epoch', i, 'iter', k, 'loss', float(loss), 'n_err', n_err)
    print('min_test_loss', min_test_loss, 'test_loss', sum_loss / testloader.num_step)


    if sum_loss / testloader.num_step < min_test_loss:
        min_test_loss = sum_loss / testloader.num_step
        print('**************save model*******************')
        torch.save(model.state_dict(), '.\checkpoint\ReID_HardModel{}.pt'.format(str(i)))
    writer.savetomat()
