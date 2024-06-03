import torch
import numpy as np
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, act_fn, c_in, c_out, subsample):
        super().__init__()
        self.act_fn = act_fn
        self.c_out = c_out
        self.subsample = subsample
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(3, 3),
                      stride=(1, 1) if not subsample else (2, 2),
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=(3, 3),
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(c_out),
        )
        if subsample:
            self.subsample_layer = nn.Conv2d(c_in, c_out, kernel_size=(1, 1),
                                             # padding=1,
                                             stride=(2, 2), bias=False)

    def forward(self, x):
        z = self.net(x)
        if self.subsample:
            x = self.subsample_layer(x)
        x = self.act_fn()(x + z)
        return x


class PreActResNetBlock(nn.Module):
    def __init__(self, act_fn, c_in, c_out, subsample):
        super().__init__()
        self.act_fn = act_fn
        self.c_out = c_out
        self.subsample = subsample
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=(3, 3),
                      padding=1,
                      stride=(1, 1) if not subsample else (2, 2),
                      bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=(3, 3),
                      padding=1,
                      bias=False),
        )
        if subsample:
            self.subsample_layer = nn.Sequential(
                nn.BatchNorm2d(c_in),
                act_fn(),
                nn.Conv2d(c_in, c_out, kernel_size=(1, 1),
                          # padding=1,
                          stride=(2, 2), bias=False)
            )

    def forward(self, x):
        z = self.net(x)
        if self.subsample:
            x = self.subsample_layer(x)
        return z + x


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes,
                 act_fn=nn.ReLU,
                 block_class=ResNetBlock,
                 num_blocks: tuple = (3, 3, 3),
                 c_hidden: tuple = (64, 128, 256)):
        super().__init__()
        # print(c_hidden)
        self.num_classes = num_classes
        self.act_fn = act_fn

        self.layers = []
        self._bn_layers = None
        if block_class == ResNetBlock:
            layer0 = nn.Sequential(
                nn.Conv2d(in_channels, c_hidden[0], kernel_size=(3, 3),
                          stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_hidden[0]),
                act_fn()
            )
        else:
            layer0 = nn.Conv2d(in_channels, c_hidden[0], kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=False)
        self.layers.append(layer0)
        in_planes = c_hidden[0]
        for block_idx, block_count in enumerate(num_blocks):
            temp_layers = []
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0)
                temp_layers.append(block_class(act_fn, in_planes, c_hidden[block_idx], subsample))
                in_planes = c_hidden[block_idx]
            if block_idx == len(num_blocks) - 1:
                temp_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.layers.append(nn.Sequential(*temp_layers))
        # self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        # self.layers = nn.Sequential(*self.layers)
        # self.layers = nn.ModuleList(self.layers)
        for layer in self.layers:
            self.add_module(str(len(self._modules)), layer)
        self.fc = nn.Linear(c_hidden[2], num_classes)

    def forward(self, x, return_feats=False):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        feats = x.mean(dim=(2, 3))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        if return_feats:
            return x, feats
        else:
            return x


if __name__ == "__main__":
    model = ResNet(3, 10)
    # for name, layer in list(model.named_children())[0][1].named_children():
    #     print(name)
    for name, layer in model.named_children():
        print(name)
    # x = torch.randn(2, 3, 32, 32)
    # print(model(x).shape)
    # bns = get_bn_layer(model)
