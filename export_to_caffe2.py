from torch.autograd import Variable
import torch.onnx
import torchvision
from models.mobilenet_multiway import *
import numpy as np
import torch as th
dummy_input = th.cuda.FloatTensor(np.zeros((10, 3, 128, 64)))
model = MobileNetV2().to('cuda')
model.eval()
model.load_state_dict(torch.load('.\checkpoint\\ReID_HardModel41.pt'))
