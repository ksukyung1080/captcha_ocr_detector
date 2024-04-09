import torch.nn as nn
import torch.nn.functional as F

## Input shape: (1(GrayScle),40,120)
class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3,6), padding=(1,1))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3,6), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.fcc1 = nn.Linear(640, 64)
        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.3, batch_first=True)
        self.fcc2 = nn.Linear(64, num_chars + 1) ## CTCLOSS BLANK 위해 class 개수 + 1

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = out.permute(0,3,1,2)
        out = out.view(out.size(0), out.size(1), -1) # => out.shape: (1,27,640)
        out = F.relu(self.fcc1(out))
        out = self.dropout(out)
        out, _ = self.lstm(out)
        out = self.fcc2(out)
        out = out.permute(1,0,2)

        return out