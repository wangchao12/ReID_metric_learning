import torch.nn as nn
import torch as th
import cv2
import math
import numpy as np


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )



def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_embeddings=128, input_size=128, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        self.mask = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                    self.mask.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                    self.mask.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.mask.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(kernel_size=(8, 4)))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.mask = nn.Sequential(*self.mask)

        # building classifier
        self.embedding = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_embeddings),
        )
        self._initialize_weights()
        self.embedding2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_embeddings),
        )

    def forward(self, x):
        x_mask = self.mask(x)
        mask = th.mean(x_mask, dim=1, keepdim=True)
        thresh = th.unsqueeze(th.mean(th.mean(mask, -1), -1, keepdim=True), -1)
        mask = th.where(mask > thresh, th.ones_like(mask), th.zeros_like(mask))
        mask = nn.UpsamplingBilinear2d(scale_factor=16)(mask)

        mask_fc = nn.AvgPool2d(kernel_size=(8, 4))(x_mask)
        mask_fc = mask_fc.view(-1, self.last_channel)
        mask_fc = self.embedding2(mask_fc)
        mask_fc = mask_fc / th.unsqueeze(th.norm(mask_fc, 2, -1), -1)

        x = self.features(x * mask)
        x = x.view(-1, self.last_channel)
        fc = self.embedding(x)
        fc = fc / th.unsqueeze(th.norm(fc, 2, -1), -1)
        return mask_fc, fc


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    img = cv2.imread('E:\Person_ReID\ReID_metric_learning\\0000_c1s1_000151_01.jpg')
    img = img.astype(np.int8)
    img_r = cv2.resize(img[:, :, 0], (128, 64))
    print()