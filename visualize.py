from models.mobilenet_multiway2 import MobileNetV2, VisualContainer
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
            output = self.model(th.cuda.FloatTensor(img_i))
            mask_img = output[-1].detach().cpu().numpy()
            mask_img = np.transpose(mask_img[0, :, :, :], axes=[1, 2, 0])
            fileName = os.path.join(self.output, name_i[0:-4] + '.jpg')
            cv2.imwrite(fileName, mask_img)




if __name__ == '__main__':
    model = MobileNetV2().to('cuda')
    model.eval()
    model.load_state_dict(th.load('.\checkpoint\\\ReID_HardModel56.pt'))
    input = 'E:\Person_ReID\ReID_metric_learning\\visualize\\test\\'
    output = 'E:\Person_ReID\ReID_metric_learning\\visualize\\mask_test2\\'
    visualizer = VisualizeMask(input, output, model)
    visualizer.save_img()
