from models.mobilenet_multiway import MobileNetV2
import os, cv2
import numpy as np
import torch as th


class VisualizeMask(object):

    def __init__(self, input, output, model):
        self.input = input
        self.output = output
        self.model = model


    def save_img(self):
        self.model.eval()
        imgs = [np.expand_dims(np.transpose(cv2.imread(os.path.join(self.input, i)), [2, 0, 1]), 0) for i in os.listdir(self.input[0:-1])[0:-1]]
        files = [i for i in os.listdir(self.input)[0:-1]]
        for img_i, name_i in zip(imgs, files):
            _, mask_img = self.model(th.cuda.FloatTensor(img_i))
            mask_img = np.transpose(mask_img.detach().cpu().numpy(), [0, 2, 3, 1])
            for id, mask_i in enumerate(mask_img):
              fileName = os.path.join(self.output, name_i[0:-4] + '_' + str(id) + '.jpg')
              cv2.imwrite(fileName, mask_i)




if __name__ == '__main__':
    model = MobileNetV2().to('cuda')
    model.load_state_dict(th.load('.\checkpoint\\\ReID_HardModel26.pt'))
    input = 'E:\Person_ReID\DataSet\DukeMTMC-reID\DukeMTMC-reID\\test_128_64\\'
    output = 'E:\Person_ReID\DataSet\DukeMTMC-reID\DukeMTMC-reID\\test_128_64_mask\\'
    visualizer = VisualizeMask(input, output, model)
    visualizer.save_img()
