import random

import torchaudio
import torch
import os
from torchaudio.transforms import Resample
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, random_split, Dataset
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
import os


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
print("noise_files count:", len(noise_files))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


print(f"Dataset size: {len(train_loader)}")

SPEECH_LEN = 16000
NOISE_HZ = 44100
NOISE_FULL = 220500

#add noise to the speech with rms level 0, 10 or 50:
def add_rms_noise(speech, noise_level):
    if noise_level == 0:
        if speech.shape[1] < SPEECH_LEN:
            speech = F.pad(speech, (0, SPEECH_LEN - speech.shape[1]))
        speech_rms = torch.sqrt(torch.mean(speech**2))
        noise_rms = torch.zeros(speech_rms.shape)
        return speech , speech_rms, noise_rms
    #pick a random noise file from the noise dataset, load it, and resample it to 16kHz
    random_noise = random.choice(noise_files)
    noise , sr = torchaudio.load(random_noise)
    resample = Resample(orig_freq=sr, new_freq=16000)
    noise_16k = resample(noise)
    if speech.shape[1] < SPEECH_LEN:
        speech = F.pad(speech, (0, SPEECH_LEN - speech.shape[1]))

    #Shorten the noise to match the speech length if it's longer, or pad it if it's shorter
    if noise_16k.shape[1] >= speech.shape[1]:
        start = random.randint(0, noise_16k.shape[1] - speech.shape[1])
        noise_16k_1s = noise_16k[:, start:start+speech.shape[1]]
    else:
        noise_16k_1s = F.pad(noise_16k, (0, speech.shape[1] - noise_16k.shape[1]))
    
    #adding noise to the speech with the specified noise level
    speech_rms = torch.sqrt(torch.mean(speech**2))
    noise_rms = torch.sqrt(torch.mean(noise_16k_1s**2))
    
    scale = (noise_level * speech_rms) / (noise_rms + 1e-10)
    scaled_noise = noise_16k_1s * scale
    noisy_speech = speech + scaled_noise
    return noisy_speech, speech_rms, noise_rms

all_labels = []


#get all labels and create a list of unique labels, then assign an id to each label based on its index in the list of unique labels
for i in range(len(train_dataset)):
    all_labels.append(train_dataset[i][2])

labels = sorted(set(all_labels))

print("one label", all_labels[0],"length" ,len(all_labels))
class NoisyDataset(Dataset):
    def __init__(self, dataset, noise_level):
        self.dataset = dataset
        self.noise_level = noise_level

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        speech,sr, label, _, _  = self.dataset[idx]
        noisy_speech, _, _ = add_rms_noise(speech, self.noise_level)
        label_id = 0 
        for i, label_name in enumerate(labels):
            if label_name == label:
                label_id = i
                break
        return noisy_speech, label_id
    

train_noise_0 = NoisyDataset(train_dataset, noise_level=0.0)
"""train_noise_10 = NoisyDataset(train_dataset, noise_level=0.1)
train_noise_50 = NoisyDataset(train_dataset, noise_level=0.5)

val_noise_0 = NoisyDataset(val_dataset, noise_level=0.0)
val_noise_10 = NoisyDataset(val_dataset, noise_level=0.1)
val_noise_50 = NoisyDataset(val_dataset, noise_level=0.5)

test_noise_0 = NoisyDataset(test_dataset, noise_level=0.0)
test_noise_10 = NoisyDataset(test_dataset, noise_level=0.1)
test_noise_50 = NoisyDataset(test_dataset, noise_level=0.5)
"""


train_noise_loader_0 = DataLoader(train_noise_0, batch_size=1, shuffle=True) #speech and labels with noise level 0.0
""" 

train_noisy_loader_10 = DataLoader(train_noisy_10, batch_size=1, shuffle=True)
train_noisy_loader_50 = DataLoader(train_noisy_50, batch_size=1, shuffle=True)

val_noisy_loader_0 = DataLoader(val_noisy_0, batch_size=1, shuffle=False)
val_noisy_loader_10 = DataLoader(val_noisy_10, batch_size=1, shuffle=False)
val_noisy_loader_50 = DataLoader(val_noisy_50, batch_size=1, shuffle=False)

test_noisy_loader_0 = DataLoader(test_noisy_0, batch_size=1, shuffle=False)
test_noisy_loader_10 = DataLoader(test_noisy_10, batch_size=1, shuffle=False)
test_noisy_loader_50 = DataLoader(test_noisy_50, batch_size=1, shuffle=False)
 """
