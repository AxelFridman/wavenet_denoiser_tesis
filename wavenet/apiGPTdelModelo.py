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
import time
import pickle
import os
import csv
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torchaudio.transforms as transformsaudio
import datetime

import torchaudio
from torchaudio.transforms import Resample
from pydub import AudioSegment

from pathlib import Path

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

#import losses
import discriminator
#import datasetParaTest
import generator
import wavenet
import postnet
from flask import Flask, request, jsonify
import torch
from pathlib import Path
from torchaudio.transforms import Resample

def load_model(model_path):
    model = torch.load(model_path)
    model.to("cpu")
    model.eval()
    return model


app = Flask(__name__)

# Load the model
model_path = "path/to/your_model.pth"
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get audio file from the request
        audio_file = request.files['audio'].read()
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_file))

        # Resample the audio
        target_sample_rate = 16000
        resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)

        # Preprocess the input
        model_input = waveform.unsqueeze(0)[:, :, :2240000].to("cpu")

        # Perform inference
        with torch.no_grad():
            model_output = model(model_input)

        cleaned_audio = model_output[0].to("cpu")[0]

        # Convert to bytes and send as response
        cleaned_audio_bytes = torchaudio.save(io.BytesIO(), cleaned_audio, target_sample_rate)
        return cleaned_audio_bytes.getvalue()

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
