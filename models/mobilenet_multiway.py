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
        self.sub_branch1 = [InvertedResidual(32, 64, 1, 6),
                            InvertedResidual(64, 96, 2, 6),
                            InvertedResidual(96, 160, 1, 6),
                            InvertedResidual(160, 320, 1, 6),
                            conv_1x1_bn(320, self.last_channel),
                            nn.AvgPool2d((4, 4))]
        ##
        self.sub_branch2 = [InvertedResidual(32, 64, 1, 6),
                             InvertedResidual(64, 96, 2, 6),
                             InvertedResidual(96, 160, 1, 6),
                             InvertedResidual(160, 320, 1, 6),
                             conv_1x1_bn(320, self.last_channel),
                             nn.AvgPool2d((4, 4))]

        # make it nn.Sequential
        self.backbone = nn.Sequential(*self.backbone)
        self.global_branch = nn.Sequential(*self.global_branch)
        self.sub_branch1 = nn.Sequential(*self.sub_branch1)
        self.sub_branch2 = nn.Sequential(*self.sub_branch2)

        # building classifier
        self.global_embedding = nn.Sequential(
           nn.Linear(self.last_channel, n_embeddings)
        )
        self.global_classifier = nn.Sequential(
            nn.Linear(self.last_channel, self.n_persons)
        )
        self.sub1_embedding = nn.Sequential(
            nn.Linear(self.last_channel, n_embeddings)
        )
        self.sub1_classifier = nn.Sequential(
            nn.Linear(self.last_channel, self.n_persons)
        )
        self.sub2_embedding = nn.Sequential(
            nn.Linear(self.last_channel, n_embeddings)
        )
        self.sub2_classifier = nn.Sequential(
            nn.Linear(self.last_channel, self.n_persons)
        )

    def forward(self, img):
        back_fm = self.backbone(img)
        global_fc = self.global_branch(back_fm)
        global_fc = global_fc.view(-1, self.last_channel)
        sub1_fc = self.sub_branch1(back_fm[:, :, 0:8, :])
        sub1_fc = sub1_fc.view(-1, self.last_channel)
        sub2_fc = self.sub_branch2(back_fm[:, :, 9:16, :])
        sub2_fc = sub2_fc.view(-1, self.last_channel)

        global_emb = self.global_embedding(global_fc)
        global_emb = global_emb / th.unsqueeze(th.norm(global_emb, 2, -1), -1)
        global_cls = self.global_classifier(global_fc)

        sub1_emb = self.sub1_embedding(sub1_fc)
        sub1_emb = sub1_emb / th.unsqueeze(th.norm(sub1_emb, 2, -1), -1)
        sub1_cls = self.sub1_classifier(sub1_fc)

        sub2_emb = self.sub2_embedding(sub2_fc)
        sub2_emb = sub2_emb / th.unsqueeze(th.norm(sub2_emb, 2, -1), -1)
        sub2_cls = self.sub2_classifier(sub2_fc)

        all_emb = th.cat((global_emb, sub1_emb, sub2_emb), dim=-1)
        all_emb = all_emb / th.unsqueeze(th.norm(all_emb, 2, -1), -1)



        return global_emb, global_cls, sub1_emb, sub1_cls, sub2_emb, sub2_cls, all_emb

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