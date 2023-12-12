print("Importando postnet")
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
class PostNetSimple(pl.LightningModule):
    def __init__(self, postnetParams, device):
        super(PostNetSimple, self).__init__()
        layers = postnetParams["layers"]
        channels = postnetParams["channels"]
        ks =postnetParams["kernel_size"]
        use_batch_norm_Post = postnetParams["use_batch_norm_Post"]
        use_dropout_Post = postnetParams["use_dropout_Post"]
        
        paddingLen = int((ks-1)/2)
        self.use_batch_norm_Post = use_batch_norm_Post
        self.use_dropout_Post = use_dropout_Post

        self.batchNormParams = nn.BatchNorm1d(channels)
        self.dropOut = nn.Dropout(use_dropout_Post)

        self.convInicial = nn.Conv1d(in_channels=1,
                                out_channels=channels,
                                kernel_size=ks, stride=1,
                                dilation=1, padding=paddingLen, bias=True).to(device)
        self.convFinal = nn.Conv1d(in_channels=channels,
                        out_channels=1,
                        kernel_size=ks, stride=1,
                        dilation=1, padding=paddingLen, bias=True).to(device)
        self.totalLayers = layers
        self.convs = []
        for conv in range(0, layers):
            self.convs.append(
                nn.Conv1d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=ks, stride=1,
                                dilation=1, padding=paddingLen, bias=True).to(device)
            )

        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.convInicial(x)
        x = self.tan(x)
        for conv in self.convs:
            x = conv(x)
            x = self.tan(x)
            if(self.use_batch_norm_Post):
                x = self.batchNormParams(x)
            x = self.dropOut(x)

        x = self.convFinal(x)
        x = self.tan(x)
        return x

