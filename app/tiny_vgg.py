from torch.nn import Module, Sequential, Linear, Dropout, ReLU, Flatten
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d
from torch.nn import init


class TinyVGG_V1(Module):
    def __init__(self, input_channels: int, output_classes: int):
        super().__init__()
        #size = (128, 128)
        self.conv_block_1 = Sequential(
            Conv2d(in_channels=input_channels,
                   out_channels=16,
                    kernel_size=3,
                    padding=1,
                    stride=1),
            BatchNorm2d(16),
            ReLU(),
            Conv2d(in_channels=16,
                   out_channels=16,
                    kernel_size=3,
                    padding=1,
                    stride=1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2),
            Dropout(0.3)
        )
        #size = (64, 64)
        self.conv_block_2 = Sequential(
            Conv2d(in_channels=16,
                   out_channels=32,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(32),
            ReLU(),
            Conv2d(in_channels=32,
                   out_channels=32,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2),
            Dropout(0.3)
        )
        #size = (32, 32)
        self.conv_block_3 = Sequential(
            Conv2d(in_channels=32,
                   out_channels=64,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64,
                   out_channels=64,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2),
            Dropout(0.3)
        )
        #size = (16, 16)
        self.classifier = Sequential(
            Flatten(),
            Linear(in_features=64*16*16,
                   out_features=128),
            ReLU(),
            Dropout(0.5),
            Linear(in_features=128,
                   out_features=output_classes),
        )
        
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.classifier(x)
        return x
        