print("Importando Losses")
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
#import tensorflow as tf
import io
from PIL import Image
#from prettytable import PrettyTable
import json




class MelspectogramLoss(nn.Module):
    def __init__(self, mel_transform1, mel_transform2, device, max_weight, epsilon, min_db_precision):
        super(MelspectogramLoss, self).__init__()
        self.epsilon = epsilon
        self.min_db_precision = min_db_precision
        self.mel_transform1 = mel_transform1
        self.mel_transform2 = mel_transform2
        self.device = device
        self.max_weight = max_weight

    def prepareSpectogram(self, melspect):
        melspect = (melspect - melspect.min() + self.epsilon) #/ (melspect.max()-melspect.min()) + 0.00000001)
        melspect=melspect.log10() * 10
        min_db = melspect.max()- self.min_db_precision
        melspect = melspect.clamp(min=min_db)
        return melspect
    
    def forward(self, y_pred, true_y):
        mel1true_y = (self.mel_transform1(true_y.to(self.device)))
        mel1y_pred = (self.mel_transform1(y_pred.to(self.device)))
        mel2true_y = (self.mel_transform2(true_y.to(self.device)))
        mel2y_pred = (self.mel_transform2(y_pred.to(self.device)))

        mel1true_y = self.prepareSpectogram(mel1true_y)
        mel1y_pred = self.prepareSpectogram(mel1y_pred)
        mel2true_y = self.prepareSpectogram(mel2true_y)
        mel2y_pred = self.prepareSpectogram(mel2y_pred)
        
        dif1 = (mel1true_y - mel1y_pred)**2
        dif2 = (mel2true_y - mel2y_pred)**2

        # Create a linear weight vector
        num_bins = dif1.size(1)
        weights = torch.linspace(1, self.max_weight, num_bins).to(self.device)

        # Apply the weights to the squared differences
        dif1 *= weights.unsqueeze(0)
        dif2 *= weights.unsqueeze(0)

        dif1 = (dif1).mean()
        dif2 = (dif2).mean()
        return dif1 , dif2
    
class LogarithmicLoss(nn.Module):
    def __init__(self):
        super(LogarithmicLoss, self).__init__()

    def forward(self, y_pred, true_y):
        #y_pred = (y_pred**2 +0.000001).log10()
        #true_y = (true_y**2 +0.000001).log10()
        whereNoSoundTrue = (true_y**2) < 0.01
        return ((whereNoSoundTrue*y_pred)**2).mean()

        #return ((y_pred - true_y)**2).mean()

class LackAmplitudeLoss(nn.Module):
    def __init__(self):
        super(LackAmplitudeLoss, self).__init__()

    def forward(self, y_pred, true_y):
        whereSoundTrue = (true_y**2) > 0.01
        return ((whereSoundTrue*(true_y - y_pred)**2)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, mel_transform1, mel_transform2, device, max_weight, epsilon, min_db_precision):
        super(CombinedLoss, self).__init__()
        self.mel_loss = MelspectogramLoss(mel_transform1, mel_transform2, device, max_weight, epsilon, min_db_precision)
        self.l1_loss = nn.L1Loss()

        self.lack_amplitude = LackAmplitudeLoss()
        self.log_loss = LogarithmicLoss()

    def forward(self, y_pred, true_y):
        #customLoss = self.custom_loss(y_pred, true_y)*1
        prop = (((y_pred**2) / ((true_y**2)+0.00001)) - 1)**2
        max_limit = 10000
        prop = torch.clamp(prop, max=max_limit) 
        customLoss = prop.mean()
        melLoss1, melLoss2 = self.mel_loss(y_pred, true_y)
        l1Loss =  self.l1_loss(y_pred, true_y)
        lackAmplitudeLoss = self.lack_amplitude(y_pred, true_y)
        logLoss = self.log_loss(y_pred, true_y)
        return  melLoss1, melLoss2, customLoss, l1Loss, lackAmplitudeLoss, logLoss
