import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

class OriNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, class_=10, ori_class=66):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """
        super(OriNet, self).__init__()
        features = init_features
        self.encoder1 = OriNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = OriNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = OriNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = OriNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = OriNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4_1 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4_1 = OriNet._block((features * 8) * 2, features * 8, name="dec4_1")
        self.upconv3_1 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3_1 = OriNet._block((features * 4) * 2, features * 4, name="dec3_1")
        self.upconv2_1 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2_1 = OriNet._block((features * 2) * 2, features * 2, name="dec2_1")
        self.upconv1_1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1_1 = OriNet._block(features * 2, features, name="dec1_1")

        self.dist = nn.Conv3d(
            in_channels=features, out_channels=class_, kernel_size=1
        )
        self.ori = nn.Conv3d(
            in_channels=features, out_channels=ori_class, kernel_size=1
        )

        self.seg = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
    def forward(self, x):
        #encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        #seg decoder
        dec4_1 = self.upconv4_1(bottleneck)
        dec4_1 = torch.cat((dec4_1, enc4), dim=1)
        dec4_1 = self.decoder4_1(dec4_1)
        dec3_1 = self.upconv3_1(dec4_1)
        dec3_1 = torch.cat((dec3_1, enc3), dim=1)
        dec3_1 = self.decoder3_1(dec3_1)
        dec2_1 = self.upconv2_1(dec3_1)
        dec2_1 = torch.cat((dec2_1, enc2), dim=1)
        dec2_1 = self.decoder2_1(dec2_1)
        dec1_1 = self.upconv1_1(dec2_1)
        dec1_1 = torch.cat((dec1_1, enc1), dim=1)
        dec1_1 = self.decoder1_1(dec1_1)
        seg = self.seg(dec1_1)
        dist = self.dist(dec1_1)
        oriention=self.ori(dec1_1)

        return seg,dist,oriention

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(inplace=True)),
                ]
            )
        )


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 64
    x = torch.Tensor(1, 1, image_size, image_size, image_size)
    x = x.to(device)
    print("x size: {}".format(x.size()))

    model = UNet3D( in_channels=1, out_channels=1, init_features=64, class_=10, ori_class=66).to(device)
    print('model',model)
    out1, out2 ,out3= model(x)
    print("out size: {}".format(out1.size()))
    print("out size: {}".format(out2.size()))
    print("out size: {}".format(out3.size()))