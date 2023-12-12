print("Ejecutando archivo entrenamiento")
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

def read_params(params):

    nameOfRun = params["nameOfRun"]
    hardwareAndSize = params["hardwareAndSize"]
    fileLocations= params["fileLocations"]
    noiseConfigs= params["noiseConfigs"]
    wavenetParams = params["wavenetParams"]
    postnetParams = params["postnetParams"]
    trainingHyperParameters = params["trainingHyperParameters"]
    savingAndLogging = params["savingAndLogging"]
    lossesConfig = params["lossesConfig"]
    weightsOfLosses= params["weightsOfLosses"]

    return (nameOfRun, hardwareAndSize , 
            fileLocations, noiseConfigs, 
            wavenetParams, postnetParams, 
            trainingHyperParameters, 
            savingAndLogging, lossesConfig, 
            weightsOfLosses)



def generateMelTransform(sample_rate, melspecConfig, device):

    mel_transform = transformsaudio.MelSpectrogram(sample_rate = sample_rate,
                                                    n_fft = melspecConfig["n_fft"],
                                                    n_mels = melspecConfig["n_mels"],
                                                    hop_length = melspecConfig["hop_length"]).to(device)
    
    return mel_transform



def doOneRun(params):


    (nameOfRun, hardwareAndSize , 
            fileLocations, noiseConfigs, 
            wavenetParams, postnetParams, 
            trainingHyperParameters, 
            savingAndLogging, lossesConfig, 
            weightsOfLosses) = read_params(params)
    device = hardwareAndSize["device"]
    sample_rate = hardwareAndSize["sample_rate"]
    inputSize = hardwareAndSize["inputSize"]
    batch_size = hardwareAndSize["batch_size"]
    wavenetOfModel = wavenet.WaveNet(wavenetParams , device)
    postnetOfModel = postnet.PostNetSimple(postnetParams, device)
    discriminatorOfModel = discriminator.Discriminator(device)

    melspec1 = 0#generateMelTransform(sample_rate, lossesConfig["melspec1Config"], device)
    melspec2 = 0#generateMelTransform(sample_rate, lossesConfig["melspec2Config"], device)

    loss_Fun = 0#losses.CombinedLoss(melspec1, melspec2, device, weightsOfLosses["max_weight_short_pitch"], lossesConfig["epsilon"], lossesConfig["min_db_precision"])

    modeloNombre = nameOfRun+'.pth'

    #device = torch.device('cpu')
    model_path = "/Users/afridman/wavenet_denoiser_tesis/limpiaModeloCpuNuevo"
    #model = torch.load(model_path)
    #model = torch.load(model_path, map_location=lambda storage, loc: storage)
        
    file = open(model_path,'rb')
    model = pickle.load(file)
    file.close()
    model.to("cpu")
   
    #model.load_state_dict(torch.load("/home/afridman/pretrained_weights.pth"))



    print("Importadas todas las librerias y modelo")
    
    
    
    model.eval()
    input_dir = Path('/Users/afridman/wavenet_denoiser_tesis/audiosSucios')
    output_dir = Path('/Users/afridman/wavenet_denoiser_tesis/audiosLimpios')
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in input_dir.glob('*.wav'):
        waveform, sample_rate = torchaudio.load(audio_path)
        target_sample_rate = 16000

        # Resample the audio
        resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        print(waveform.shape)
        waveform = torch.mean(waveform, dim=0, keepdim=True) # Hacer mono canal
        print(waveform.shape)

        waveform = resample_transform(waveform)
        print(waveform.shape)

        model_input = waveform.unsqueeze(0)[:, :, :180000].to("cpu")#preprocess(waveform)
        print(waveform.shape)

        print(model_input.shape)
        with torch.no_grad():
            model_output = model(model_input)

        cleaned_audio = (model_output[0]).to("cpu")[0]
        print(cleaned_audio.shape)
        torchaudio.save(output_dir / audio_path.name, cleaned_audio, target_sample_rate)
def saveResults(data, directoryBase):
    # Define the CSV file path

    # Specify the columns
    columns = ['customLoss', 'lackAmplitudeLoss', 'logLoss', 'lossAud', 'lossMel1', 'lossMel2', 'pesq', 'snr', 'srmr', 'stoi', 'val_loss']

    # Specify the file name and path
    file_path = directoryBase + '/results.csv'

    # Check if the file already exists
    if not os.path.isfile(file_path):
        # Create a dataframe with empty values
        df = pd.DataFrame(columns=columns)
        
        # Save the dataframe to a CSV file
        df.to_csv(file_path, index=False)
        print(f"CSV file '{file_path}' created successfully!")
    else:
        print(f"CSV file '{file_path}' already exists.")
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Append a new row to the DataFrame
    df_dictionary = pd.DataFrame([data])
    output = pd.concat([df, df_dictionary], ignore_index=True)
    # Save the updated DataFrame back to the CSV file
    output.to_csv(file_path, index=True)


nombreJson = "/Users/afridman/wavenet_denoiser_tesis/runs/runs.json"

with open(nombreJson, 'r') as file:
    runs = json.load(file)

nameFiles = runs["configRunsName"]
print(nameFiles)

for currentFile in nameFiles:
    currentFile = "/Users/afridman/wavenet_denoiser_tesis/runs/" + currentFile
    print(currentFile)

    with open(currentFile, 'r') as file:
        params = json.load(file)
    print("Working on " + currentFile)
    doOneRun(params)