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


def branch(params, layers, input_channel):
    for t, c, n, s in params:
        output_channel = int(c * 1.)
        for i in range(n):
            if i == 0:
                layers.append(InvertedResidual(input_channel, output_channel, s, t))
            else:
                layers.append(InvertedResidual(input_channel, output_channel, 1, t))
            input_channel = output_channel
    return layers


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
    def __init__(self, n_embeddings=128, input_size=128, width_mult=1., n_person=4100):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.n_embeddings = n_embeddings
        self.n_persons = n_person
        self.backbone_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            [6, 24, 3, 2],
            [6, 32, 6, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1024 * width_mult) if width_mult > 1.0 else 1024
        self.backbone = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.backbone_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.backbone.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.backbone.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.global_branch = [InvertedResidual(32, 64, 2, 6),
                              InvertedResidual(64, 96, 1, 6),
                              InvertedResidual(96, 160, 1, 6),
                              InvertedResidual(160, 320, 1, 6),
                              conv_1x1_bn(320, self.last_channel),
                              nn.AvgPool2d((8, 4))]
        ##
        self.sub_branch21 = [InvertedResidual(32, 64, 1, 6),
                             InvertedResidual(64, 96, 2, 6),
                             InvertedResidual(96, 160, 1, 6),
                             InvertedResidual(160, 320, 1, 6),
                             conv_1x1_bn(320, self.last_channel),
                             nn.AvgPool2d((4, 4))]
        ##
        self.sub_branch22 = [InvertedResidual(32, 64, 1, 6),
                             InvertedResidual(64, 96, 2, 6),
                             InvertedResidual(96, 160, 1, 6),
                             InvertedResidual(160, 320, 1, 6),
                             conv_1x1_bn(320, self.last_channel),
                             nn.AvgPool2d((4, 4))]
        ##
        self.sub_branch31 = [InvertedResidual(32, 64, 1, 6),
                             InvertedResidual(64, 96, 2, 6),
                             InvertedResidual(96, 160, 1, 6),
                             InvertedResidual(160, 320, 1, 6),
                             conv_1x1_bn(320, self.last_channel),
                             nn.AvgPool2d((3, 4))]
        ##
        self.sub_branch32 = [InvertedResidual(32, 64, 1, 6),
                             InvertedResidual(64, 96, 2, 6),
                             InvertedResidual(96, 160, 1, 6),
                             InvertedResidual(160, 320, 1, 6),
                             conv_1x1_bn(320, self.last_channel),
                             nn.AvgPool2d((3, 4))]
        ##
        self.sub_branch33 = [InvertedResidual(32, 64, 1, 6),
                             InvertedResidual(64, 96, 2, 6),
                             InvertedResidual(96, 160, 1, 6),
                             InvertedResidual(160, 320, 1, 6),
                             conv_1x1_bn(320, self.last_channel),
                             nn.AvgPool2d((3, 4))]

        # make it nn.Sequential
        self.backbone = nn.Sequential(*self.backbone)
        self.global_branch = nn.Sequential(*self.global_branch)
        self.sub_branch21 = nn.Sequential(*self.sub_branch21)
        self.sub_branch22 = nn.Sequential(*self.sub_branch22)
        self.sub_branch31 = nn.Sequential(*self.sub_branch31)
        self.sub_branch32 = nn.Sequential(*self.sub_branch32)
        self.sub_branch33 = nn.Sequential(*self.sub_branch33)
        # building classifier
        self.global_embedding = nn.Sequential(
           nn.Linear(self.last_channel, n_embeddings)
        )

        self.sub21_embedding = nn.Sequential(
            nn.Linear(self.last_channel, n_embeddings)
        )

        self.sub22_embedding = nn.Sequential(
            nn.Linear(self.last_channel, n_embeddings)
        )

        self.sub31_embedding = nn.Sequential(
            nn.Linear(self.last_channel, n_embeddings)
        )

        self.sub32_embedding = nn.Sequential(
            nn.Linear(self.last_channel, n_embeddings)
        )
        self.sub33_embedding = nn.Sequential(
            nn.Linear(self.last_channel, n_embeddings)
        )


    def forward(self, img):
        back_fm = self.backbone(img)
        global_fc = self.global_branch(back_fm)
        global_fc = global_fc.view(-1, self.last_channel)
        sub21_fc = self.sub_branch21(back_fm[:, :, 0:7, :])
        sub21_fc = sub21_fc.view(-1, self.last_channel)

        sub22_fc = self.sub_branch22(back_fm[:, :, 8:15, :])
        sub22_fc = sub22_fc.view(-1, self.last_channel)

        sub31_fc = self.sub_branch31(back_fm[:, :, 0:5, :])
        sub31_fc = sub31_fc.view(-1, self.last_channel)

        sub32_fc = self.sub_branch32(back_fm[:, :, 5:10, :])
        sub32_fc = sub32_fc.view(-1, self.last_channel)

        sub33_fc = self.sub_branch33(back_fm[:, :, 10:15, :])
        sub33_fc = sub33_fc.view(-1, self.last_channel)

        global_emb = self.global_embedding(global_fc)
        global_emb = global_emb / th.unsqueeze(th.norm(global_emb, 2, -1), -1)

        sub21_emb = self.sub21_embedding(sub21_fc)
        sub21_emb = sub21_emb / th.unsqueeze(th.norm(sub21_emb, 2, -1), -1)

        sub22_emb = self.sub22_embedding(sub22_fc)
        sub22_emb = sub22_emb / th.unsqueeze(th.norm(sub22_emb, 2, -1), -1)

        sub31_emb = self.sub31_embedding(sub31_fc)
        sub31_emb = sub31_emb / th.unsqueeze(th.norm(sub31_emb, 2, -1), -1)

        sub32_emb = self.sub32_embedding(sub32_fc)
        sub32_emb = sub32_emb / th.unsqueeze(th.norm(sub32_emb, 2, -1), -1)

        sub33_emb = self.sub33_embedding(sub33_fc)
        sub33_emb = sub33_emb / th.unsqueeze(th.norm(sub33_emb, 2, -1), -1)

        all_emb = th.cat((global_emb, sub21_emb, sub22_emb, sub31_emb, sub32_emb, sub33_emb), dim=-1)
        all_emb = all_emb / th.unsqueeze(th.norm(all_emb, 2, -1), -1)

        return global_emb, global_fc, sub21_emb, sub21_fc, sub22_emb, sub22_fc,  \
               sub31_emb, sub31_fc, sub32_emb, sub32_fc,  sub33_emb, sub33_fc, all_emb

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


class ModelContainer(nn.Module):

    def __init__(self, model):
        super(ModelContainer, self).__init__()
        self.model = model
        self.last_channel = model.last_channel
        self.n_persons = model.n_persons
        self.global_classifier = nn.Sequential(
            nn.Linear(self.last_channel, self.n_persons)
        )
        self.sub21_classifier = nn.Sequential(
            nn.Linear(self.last_channel, self.n_persons)
        )
        self.sub22_classifier = nn.Sequential(
            nn.Linear(self.last_channel, self.n_persons)
        )
        self.sub31_classifier = nn.Sequential(
            nn.Linear(self.last_channel, self.n_persons)
        )
        self.sub32_classifier = nn.Sequential(
            nn.Linear(self.last_channel, self.n_persons)
        )
        self.sub33_classifier = nn.Sequential(
            nn.Linear(self.last_channel, self.n_persons)
        )

    def forward(self, input):
        global_emb, global_fc, sub21_emb, sub21_fc, sub22_emb, sub22_fc, \
        sub31_emb, sub31_fc, sub32_emb, sub32_fc, sub33_emb, sub33_fc, all_emb = self.model(input)
        global_cls = self.global_classifier(global_fc)
        sub21_cls = self.sub21_classifier(sub21_fc)
        sub22_cls = self.sub22_classifier(sub22_fc)
        sub31_cls = self.sub31_classifier(sub31_fc)
        sub32_cls = self.sub32_classifier(sub32_fc)
        sub33_cls = self.sub33_classifier(sub33_fc)

        return global_emb, global_cls, sub21_emb, sub21_cls, sub22_emb, sub22_cls,\
               sub31_emb, sub31_cls, sub32_emb, sub32_cls, sub33_emb, sub33_cls, all_emb





if __name__ == '__main__':
    img = cv2.imread('E:\Person_ReID\ReID_metric_learning\\0000_c1s1_000151_01.jpg')
    img = img.astype(np.int8)
    img_r = cv2.resize(img[:, :, 0], (128, 64))
    print()