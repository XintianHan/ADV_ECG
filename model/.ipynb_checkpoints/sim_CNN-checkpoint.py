import torch.nn as nn
import torch
import math

class CNN(nn.Module):

    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 320, kernel_size=24, stride=1, padding=11, bias=True)
        self.bn1 = nn.BatchNorm1d(320)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation = 2)

    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size,
                            stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x, bounds=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer0(x)
        # x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
        x = torch.max(x,2)[0]
        x = x.view(x.size(0), -1)
        x = self.fc(x)

#         x = self.conv_merge(x)
#         x = torch.squeeze(x, dim=2)
#         x = self.vlp(x, bounds=bounds)

        return x