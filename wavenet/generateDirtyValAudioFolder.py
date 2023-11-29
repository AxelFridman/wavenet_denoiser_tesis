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
import tensorflow as tf
import io
from PIL import Image
from prettytable import PrettyTable
import json
import dataset

directoryBase = "/home/afridman"
locationValidationFile = "/wavenet/CSV/newCSV/audiosVal.csv"
maxRuido = 0.05
useDistortions = True
inputSize = 32000
snr = 1
add_impulse_response = True
DificultadReverb = 0.6
stepsTillDificultyIncrement = []
maxRuidoChange = []
SNRChange = []
ReverbChange = []
sample_rate = 16000

valdataset = dataset.AudioCleaningDataset(directoryBase+locationValidationFile, 
                                directoryBase+'/extra/audiosPaises', 
                                directoryBase+'/extra/ruidosDivididos',
                                directoryBase+"/wavenet/CSV/ruido_validation.csv",
                                directoryBase+"/extra/irDivididos/irval",
                                target_length=inputSize,
                                max_noise=maxRuido, use_distortions=useDistortions, 
                                fixed_interval=True, snr=snr,
                                ir_on=add_impulse_response, reverb_difficulty_level=DificultadReverb,
                                steps_till_difficulty_increment=stepsTillDificultyIncrement,
                                max_noise_change=maxRuidoChange,
                                snr_change=SNRChange,
                                reverb_change=ReverbChange, sample_rate=sample_rate)

valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=16)


for waveform, label, audio_file_name in valdataloader:
    audio_file_name = audio_file_name[0]
    audio_file_name = audio_file_name.split(sep="/")[-1]
    
    # Assuming 'directoryBase' and 'sample_rate' are defined
    sf.write(directoryBase + "/extra/suciosVal/"+audio_file_name, waveform[0][0], sample_rate, 'PCM_24')
