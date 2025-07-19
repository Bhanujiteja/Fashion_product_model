import torch.nn as nn

class MultiOutputModel(nn.Module):
    def __init__(self, num_baseColour, num_masterCategory, num_gender, num_usage):
        super(MultiOutputModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(16 * 64 * 64, 512)

        self.baseColour = nn.Linear(512, num_baseColour)
        self.masterCategory = nn.Linear(512, num_masterCategory)
        self.gender = nn.Linear(512, num_gender)
        self.usage = nn.Linear(512, num_usage)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        return {
            'baseColour': self.baseColour(x),
            'masterCategory': self.masterCategory(x),
            'gender': self.gender(x),
            'usage': self.usage(x)
        }
