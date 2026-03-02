import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, random_split
import torchvision

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
import os

PATH_ESC50 = "./noise_audio"

noise_files = []

for i in os.listdir(PATH_ESC50):
    noise_files.append(i)

dataset = DataLoader(noise_files, batch_size=1, shuffle=True)
lengths = [int(len(dataset)*0.7), int(len(dataset)*0.15), len(dataset)-int(len(dataset)*0.7)-int(len(dataset)*0.15)]

train_dataset_noise, val_dataset_noise, test_dataset_noise = random_split(dataset, lengths)

print(f"Dataset size: {len(train_dataset_noise)}")
print(f"Dataset size: {len(val_dataset_noise)}")
print(f"Dataset size: {len(test_dataset_noise)}")
print(f"Dataset size: {len(train_dataset)}")
print(f"Dataset size: {len(val_dataset)}")
print(f"Dataset size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)