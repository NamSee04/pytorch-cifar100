import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate

        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squeeze_channel):
        super(Fire, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squeeze_channel, kernel_size=1),
            nn.BatchNorm2d(squeeze_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channel, out_channel // 2, kernel_size=1),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channel, out_channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([self.expand_1x1(x), self.expand_3x3(x)], 1)
        return x

class SqueezeNetv2(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=952):

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(64, 96, 16),
            DenseBlock(in_channels=96, growth_rate=32, num_layers=2),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(160, 160, 32),
            DenseBlock(in_channels=160, growth_rate=32, num_layers=4),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(288, 288, 48),
            DenseBlock(in_channels=288, growth_rate=48, num_layers=4),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(480, 480, 64)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(480, class_num, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

def squeezenetv2(class_num=952):
    return SqueezeNetv2(class_num=class_num)

model = squeezenetv2()
print(sum(p.numel() for p in model.parameters()))
