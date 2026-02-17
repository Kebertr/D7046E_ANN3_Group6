# ANN Project: Audio Classification with CNN and RNN

## Project Overview

For students who select the ANN project track, the goal is to classify audio commands from the **Google Speech Commands dataset** using both Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), and compare their robustness to varying levels of background "noise" sampled from another dataset.

## Datasets

### Google Speech Commands Dataset
The [Google Speech Commands dataset](https://pytorch.org/audio/stable/generated/torchaudio.datasets.SPEECHCOMMANDS.html) is a collection of short (1 second) audio clips of spoken words. The dataset contains 65,000 utterances of 30 short words by thousands of different speakers. Each audio file is sampled at 16 kHz.

**Dataset characteristics:**
- Sample rate: 16 kHz
- Duration: 1 second per clip
- Number of classes: 30 words (e.g., "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", etc.)
- Total samples: ~65,000

**How to access:** The dataset can be loaded directly using `torchaudio.datasets.SPEECHCOMMANDS` or downloaded from the [official repository](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

### ESC-50: Environmental Sound Classification
The [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) contains 2,000 environmental audio recordings organized into 50 classes (40 clips per class). For this project, you will use sounds from this dataset as background noise to test the robustness of your models.

**Dataset characteristics:**
- Sample rate: 44.1 kHz (needs resampling to 16 kHz before combining with a speech command)
- Duration: 5 seconds per clip
- Categories include: rain, sea waves, crackling fire, clock tick, dog bark, footsteps, door knock, etc.

**How to access:** Download from the [ESC-50 GitHub repository](https://github.com/karolpiczak/ESC-50).

## Project Task

### Objective
Compare the performance of CNN and RNN architectures for speech command classification under different levels of background noise. The models should have a similar number of trainable parameters to ensure a fair comparison.

### Requirements

#### 1. Data Preprocessing
- **Resample** the ESC-50 background sounds from 44.1 kHz to **16 kHz** to match the Google Speech Commands sample rate
- Create augmented versions of the Speech Commands dataset by mixing in background noise at different signal-to-noise ratios

#### 2. Noise Mixing
Evaluate classification accuracy on the Speech Commands dataset with **three noise levels** based on signal RMS (Root Mean Square):

- **0% noise:** Clean speech commands audio (no background noise added)
- **10% noise:** RMS of background noise = 10% of speech command RMS
- **50% noise:** RMS of background noise = 50% of speech command RMS (i.e., equal signal and noise power)

The RMS mixing ensures consistent and measurable noise levels across all experiments.

#### 3. Model Implementation
Implement and train two model architectures:

**a) Convolutional Neural Network (CNN)**
- Process audio spectrograms (e.g., Mel-spectrograms, MFCCs) as 2D images
- Use convolutional layers to extract features from time-frequency representations

**b) Recurrent Neural Network (RNN)**
- Process raw audio waveforms or frame-based features sequentially
- Use LSTM or GRU layers to capture temporal dependencies

**Important:** Both models should have approximately the same number of parameters for a fair comparison.

#### 4. Performance Metrics
For each model and noise level, report:
- Training accuracy
- Validation accuracy
- Test accuracy
- Confusion matrix

#### 5. Minimum Accuracy Requirement
Your models should achieve **at least 75% test accuracy** on the clean (0% noise) Speech Commands dataset. This ensures that your baseline model is working correctly before evaluating noise robustness.

#### 6. Analysis
Compare and discuss:
- Which architecture (CNN or RNN) performs better at each noise level?
- How does performance degrade as noise increases?
- What are the computational trade-offs between the two approaches?
- Which types of commands are most/least robust to noise?


### Optional challenge (deepening)
Can you combine CNN and RNN to make one model that performs better in terms of accuracy versus total parameter count? Which type of model performs best at very high noise level? What level of noise is too high for reliable classification?

## Key Resources

### PyTorch Documentation
- **[torchaudio.datasets.SPEECHCOMMANDS](https://pytorch.org/audio/stable/generated/torchaudio.datasets.SPEECHCOMMANDS.html)** - Official documentation for loading the Speech Commands dataset
- **[torchaudio.transforms](https://pytorch.org/audio/stable/transforms.html)** - Audio transformations including MelSpectrogram, MFCC, and Resample
- **[PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)** - For implementing RNN models
- **[PyTorch Conv2d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)** - For implementing CNN models

### Useful Libraries
- **torchaudio**: For audio loading, transformations (Mel-spectrogram, MFCC, resampling), and the SPEECHCOMMANDS dataset
- **torch.nn**: For building CNN (Conv2d, MaxPool2d) and RNN (LSTM, GRU) architectures
- **librosa**: Alternative library for audio feature extraction and processing
- **numpy**: For signal processing, RMS calculations, and noise mixing

## Getting Started

1. **Load the datasets**: Use `torchaudio.datasets.SPEECHCOMMANDS` for speech commands and download ESC-50 from GitHub
2. **Explore the data**: Understand audio formats, sample rates, and class distributions
3. **Implement preprocessing**: Create a pipeline for resampling ESC-50 sounds and mixing noise at different RMS levels
4. **Design CNN architecture**: Start with a model that processes Mel-spectrograms or MFCCs as 2D images
5. **Design RNN architecture**: Build an LSTM/GRU model with similar parameter count that processes sequential audio features
6. **Train on clean data**: Ensure both models achieve at least 75% accuracy before adding noise
7. **Evaluate on noisy data**: Test both models on 10% and 50% noise levels
8. **Analyze and compare**: Create plots and confusion matrices to compare model performance

## Submission

Your project submission should include:
- Complete Jupyter notebook with all code, visualizations, and analysis
- Clear documentation of your model architectures (number of parameters for each)
- Performance comparison plots (accuracy vs. noise level for both models)
- Confusion matrices for at least the 0%, 10% and 50% noise conditions
- Discussion of results and conclusions