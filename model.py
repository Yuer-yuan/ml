import torch


class SRCNN(torch.nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=9, padding=0)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x

    def init_weights(self):
        torch.nn.init.normal_(self.conv1.weight, mean=0, std=0.001)
        torch.nn.init.constant_(self.conv1.bias, val=0)
        torch.nn.init.normal_(self.conv1.weight, mean=0, std=0.001)
        torch.nn.init.constant_(self.conv2.bias, val=0)
        torch.nn.init.normal_(self.conv1.weight, mean=0, std=0.001)
        torch.nn.init.constant_(self.conv3.bias, val=0)
