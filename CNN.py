import torchaudio
import os
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        # Implement the forward function in the network
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

train_dataset = SPEECHCOMMANDS(
    root='./', 
    url='speech_commands_v0.01', 
    download=True,
    subset="training"
)

val_dataset = SPEECHCOMMANDS(
    root='./', 
    url='speech_commands_v0.01', 
    download=True,
    subset="validation"
)
test_dataset = SPEECHCOMMANDS(
    root='./', 
    url='speech_commands_v0.01', 
    download=True,
    subset="testing"
)

PATH_ESC50 = "./noise_audio"

noise_files = []

for i in os.listdir(PATH_ESC50):
    if os.path.exists("noise_audio/" + i):
        noise_files.append("noise_audio/" + i)


dataset = noise_files
lengths = [int(len(dataset)*0.7), int(len(dataset)*0.15), len(dataset)-int(len(dataset)*0.7)-int(len(dataset)*0.15)]
train_dataset_noise, val_dataset_noise, test_dataset_noise = random_split(dataset, lengths)

train_loader_noise = DataLoader(train_dataset_noise, batch_size=1, shuffle=True)
val_loader_noise = DataLoader(val_dataset_noise, batch_size=1, shuffle=False)
test_loader_noise = DataLoader(test_dataset_noise, batch_size=1, shuffle=False)



print(f"Dataset size: {len(train_dataset_noise)}")
print(f"Dataset size: {len(val_dataset_noise)}")
print(f"Dataset size: {len(test_dataset_noise)}")
print(f"Dataset size: {len(train_dataset)}")
print(f"Dataset size: {len(val_dataset)}")
print(f"Dataset size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)