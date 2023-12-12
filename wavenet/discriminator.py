print("Importando Discriminator")
import torch
import torch.nn as nn
import torchaudio
import torch.fft
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
import seaborn as sns
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torchaudio.transforms as transformsaudio
import datetime
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
import time
import torch.nn.functional as F
import torchaudio.functional as Fa

import math
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback

from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import SpeechReverberationModulationEnergyRatio
from torchmetrics.audio import SignalNoiseRatio

import librosa
import io
from PIL import Image
import json
class Discriminator(pl.LightningModule):
        def __init__(self, device):
            super(Discriminator, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, stride=1, groups=1).to(device)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=41, stride=4, groups=4).to(device)
            self.conv3 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=41, stride=4, groups=16).to(device)
            self.conv4 = nn.Conv1d(in_channels=256, out_channels=1024, kernel_size=41, stride=4, groups=64).to(device)
            self.conv5 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=41, stride=4, groups=256).to(device)
            self.conv6 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, groups=1).to(device)
            self.conv7 = nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, groups=1).to(device)
            self.linfinal = nn.Linear(106, 1)
            self.sigm = nn.Sigmoid()
            self.leakyrelu = nn.LeakyReLU(0.01)

            self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
            self.loss_fn = nn.BCELoss().to(device)

        def forward(self, x):
            x = self.conv1(x)
            x = self.leakyrelu(x)
            x = self.conv2(x)
            x = self.leakyrelu(x)
            x = self.conv3(x)
            x = self.leakyrelu(x)
            x = self.conv4(x)
            x = self.leakyrelu(x)
            x = self.conv5(x)
            x = self.leakyrelu(x)
            x = self.conv6(x)
            x = self.leakyrelu(x)
            x = self.conv7(x)
            x = self.linfinal(x)
            x = self.sigm(x)


            return x

        def step_train(self, audio, label):
            self.optimizer.zero_grad()

            # forward pass
            y_hat = self.forward(audio)[0][0][0]

            loss = self.loss_fn(y_hat, label)

            # backprop
            loss.backward(retain_graph=True)

            # update weights
            self.optimizer.step()

            return loss
