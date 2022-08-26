import torch
from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, out_channels=2) -> None:
        super().__init__()

        self.in_channels = 3
        self.out_channels = out_channels
        # self.avgpool = nn.AdaptiveAvgPool2d((32, 32))
        self.layers = nn.Sequential( 
            # nn.Linear(self.in_channels, 16),
            nn.ReLU(True), 
            nn.Dropout(p=0.2), 
            # nn.Linear(16, 16), 
            # nn.ReLU(True), 
            # nn.Dropout(p=0.5), 
            nn.Linear(32, self.out_channels), 
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(1)
        x, _ = nn.LSTM(3, 16, batch_first=True, bidirectional=True)(x)
        x = self.layers(x)
        x = x.squeeze(1)
        return x


def mlp_1(out_channels=2):
    return FullyConnected(out_channels=out_channels)

if __name__ == '__main__':

    model = mlp_1(out_channels=2)
    model.eval()
    x = torch.rand(16, 3)
    print(x)
    print(model(x))