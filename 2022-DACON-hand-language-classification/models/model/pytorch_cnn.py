import torch
import torch.nn

class CNNclassification(torch.nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        # 첫번째층
        # ImgIn shape=(batch_size, 28, 28, 2)
        #    Conv     -> (batch_size, 28, 28, 16)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout(p=0.3))
        
        # 두번째층
        # ImgIn shape=(batch_size, 28, 28, 16)
        #    Conv      ->(batch_size, 28, 28, 32)
        #    Pool      ->(batch_size, 9, 9, 32)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 128, kernel_size=5, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(kernel_size=3, stride=3),
            torch.nn.Dropout(p=0.3))
        
        # 세번째층
        # ImgIn shape=(batch_size, 9, 9, 32)
        #    Conv      ->(batch_size, 9, 9, 64)
        #    Pool      ->(batch_size, 3, 3, 64)        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=3, stride=3),
            torch.nn.Dropout(p=0.3))
        
        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(6272, 11, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight) # fc 가중치 초기화

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten
        out = self.fc(out)
        return out