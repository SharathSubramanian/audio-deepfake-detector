import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

        # Grad-CAM
        self.feature_maps = None

    def forward(self, x, explain=False):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.conv2(x)

        if explain:
            self.feature_maps = x
            x.retain_grad()

        x = self.pool(F.relu(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNN_Dropout(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

        # Grad-CAM
        self.feature_maps = None

    def forward(self, x, explain=False):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.conv2(x)

        if explain:
            self.feature_maps = x
            x.retain_grad()

        x = self.pool(F.relu(x))

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class CNN_Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.attention = nn.Conv2d(32, 32, 1)

        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

        self.gradients = None
        self.feature_maps = None

    def save_gradients(self, grad):
        self.gradients = grad

    def forward(self, x, explain=False):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.conv2(x)

        if explain:
            x.register_hook(self.save_gradients)

        self.feature_maps = x

        attn = torch.sigmoid(self.attention(x))
        x = x * attn

        x = self.pool(F.relu(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)