import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 8), padding=(0, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 8), padding=(0, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 8), padding=(0, 3)),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 8), padding=(0, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 8), padding=(0, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),

            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=(1, 8), padding=(0, 3)),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=False),
        )

        self.encoderLayer = nn.TransformerEncoderLayer(d_model=4, nhead=4, dim_feedforward=8)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=6)

        self.fc = nn.Sequential(
            nn.Linear(in_features=372, out_features=512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(2, 0, 1).contiguous()
        x = self.encoder(x)
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

