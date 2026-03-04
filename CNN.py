import random

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import torchaudio
import os
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchaudio.transforms import Resample
import torchaudio
# Running with gpu otherwise it takes ages
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# To enable confusion matrix to render and then continue
plt.ion()
# Our model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 5, 120)
        self.fc2 = nn.Linear(120, 30)

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

LEARNING_RATE = 1e-3

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    min_LSTM_loss = 10
    
    model.train()
    melspectogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=1024,hop_length=512,n_mels=64).to(gpu)

    for i in range(num_epochs):
        epoch_training_loss = 0
        epoch_validation_loss = 0

        model.train()
        correct = 0
        total = 0
        for waveform, targets in train_loader:
            waveform = waveform.to(gpu)
            targets = targets.to(gpu)

            mel2D = melspectogram(waveform)

            mel2D = torch.log(mel2D + 1e-9)
            # Forward pass
            prediction = model(mel2D)
            loss = criterion(prediction, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            

            guess = torch.argmax(prediction, dim=-1)
            for i in range(len(targets)):
                if guess[i].item() == targets[i].item():
                    correct += 1
                total += 1
        epoch_training_loss += loss.item()  
        print("Training accuracy: ", correct/total)
            
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for waveform, targets in val_loader:
                waveform = waveform.to(gpu)
                targets = targets.to(gpu)
                
                mel2D = melspectogram(waveform)

                mel2D = torch.log(mel2D + 1e-9)
                
                # Forward pass
                prediction = model(mel2D)
                loss = criterion(prediction, targets)
                epoch_validation_loss += loss.item()
            
                

                guess = torch.argmax(prediction, dim=-1)
                for i in range(len(targets)):
                    if guess[i].item() == targets[i].item():
                        correct += 1
                    total += 1   
        epoch_validation_loss /= len(val_loader)
        if epoch_validation_loss<min_LSTM_loss:
            min_LSTM_loss = epoch_validation_loss
            torch.save(model, 'best_model.pt')
        print("validation accuracy: ", correct/total)
                    
    return torch.load("best_model.pt")
        
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    preds = []
    lbs = []
    melspectogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=1024,hop_length=512,n_mels=64).to(gpu)

    with torch.no_grad():
        for waveform, targets in test_loader:
            waveform = waveform.to(gpu)
            targets = targets.to(gpu)

            mel2D = melspectogram(waveform)

            mel2D = torch.log(mel2D + 1e-9)

            prediction = model(mel2D)

            guess = torch.argmax(prediction, dim=-1)
            for i in range(len(targets)):
                if guess[i].item() == targets[i].item():
                    correct += 1
                total += 1
                preds.append(guess[i].item())
                lbs.append(targets[i].item())

    accuracy = correct/total

    conf_mat = confusion_matrix(lbs, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    

    disp.plot()
    plt.show()
    plt.pause(0.01)
    return accuracy



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

train_loader_noise = DataLoader(train_dataset_noise, batch_size=64, shuffle=True)
val_loader_noise = DataLoader(val_dataset_noise, batch_size=64, shuffle=False)
test_loader_noise = DataLoader(test_dataset_noise, batch_size=64, shuffle=False)


print(f"Dataset size: {len(train_dataset_noise)}")
print(f"Dataset size: {len(val_dataset_noise)}")
print(f"Dataset size: {len(test_dataset_noise)}")
print(f"Dataset size: {len(train_dataset)}")
print(f"Dataset size: {len(val_dataset)}")
print(f"Dataset size: {len(test_dataset)}")


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

print(labels)
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
    


"""train_noise_10 = NoisyDataset(train_dataset, noise_level=0.1)
train_noise_50 = NoisyDataset(train_dataset, noise_level=0.5)

val_noise_0 = NoisyDataset(val_dataset, noise_level=0.0)
val_noise_10 = NoisyDataset(val_dataset, noise_level=0.1)
val_noise_50 = NoisyDataset(val_dataset, noise_level=0.5)

test_noise_0 = NoisyDataset(test_dataset, noise_level=0.0)
test_noise_10 = NoisyDataset(test_dataset, noise_level=0.1)
test_noise_50 = NoisyDataset(test_dataset, noise_level=0.5)
"""


#train_noise_loader_0 = DataLoader(train_noise_0, batch_size=1, shuffle=True) #speech and labels with noise level 0.0
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



noise_levels = [0,0.1,0.5]

for levels in noise_levels:
    # Load our network
    model = Net().to(gpu)


    # Define our loss function
    criterion = nn.CrossEntropyLoss()

    # Define our optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    train_noise_0 = NoisyDataset(train_dataset, noise_level=levels)
    val_noise_0 = NoisyDataset(val_dataset, noise_level=levels)
    test_noise_0 = NoisyDataset(test_dataset, noise_level=levels)

    train_loader = DataLoader(train_noise_0, batch_size=64, shuffle=True)

    val_loader = DataLoader(val_noise_0, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_noise_0, batch_size=64, shuffle=False)

    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, 10)
    print(test_model(trained_model, test_loader))