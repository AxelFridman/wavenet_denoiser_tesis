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
import os
import csv
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

import losses
import discriminator
import dataset
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

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


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

    print("Importadas todas las librerias")
    
    melspec1 = generateMelTransform(sample_rate, lossesConfig["melspec1Config"], device)
    melspec2 = generateMelTransform(sample_rate, lossesConfig["melspec2Config"], device)
    
    nombreParaGuardarLog = nameOfRun#.split("/")
    #nombreParaGuardarLog = nombreParaGuardarLog[-2] + nombreParaGuardarLog[-1] 
    writer = SummaryWriter(log_dir=fileLocations["saveTensorboardLocation"] + "/"+nombreParaGuardarLog , comment=nameOfRun)
    #%load_ext tensorboard
    loss_Fun = losses.CombinedLoss(melspec1, melspec2, device, weightsOfLosses["max_weight_short_pitch"], lossesConfig["epsilon"], lossesConfig["min_db_precision"])


    fullvaldataset = dataset.AudioCleaningDataset(fileLocations["FullValLocations"], 
                                                   fileLocations["directoryBase"],
                                    target_length=inputSize, noiseConfigs=noiseConfigs, isTrain = False )


    fullValDataloader = DataLoader(fullvaldataset, batch_size=batch_size, shuffle=False, num_workers=16)

        # Initialize the sum of each metric and loss
    sum_snr, sum_pesq, sum_stoi, sum_srmr, sum_loss = 0, 0, 0, 0, 0
    # Number of batches
    num_batches = 0
    srmrf = SpeechReverberationModulationEnergyRatio(sample_rate).to(device)
    wb_pesqf = PerceptualEvaluationSpeechQuality(sample_rate, 'wb').to(device)
    stoif = ShortTimeObjectiveIntelligibility(sample_rate, False).to(device)
    snrf = SignalNoiseRatio(sample_rate).to(device)

    # Iterate over the entire validation set
    for val_batch in fullValDataloader:
        X, y , audio_file_name, genero, pais, texto= val_batch
        X = X.to(device)
        y = y.to(device)


        lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss = loss_Fun(X, y)
        lossDiscriminador = 0
        loss = calculateTotalLoss(weightsOfLosses,lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss, lossDiscriminador)
        
        # compute metrics
        snr = snrf(X,y)#compute_snr(y, y_pred)
        pesq = wb_pesqf(X,y)#compute_pesq(y, y_pred)
        stoi = stoif(X,y)#compute_stoi(y, y_pred)
        srmr = srmrf(X)#compute_srmr(y, y_pred)

        # Add metrics and loss to the corresponding sum
        sum_snr += snr
        sum_pesq += pesq
        sum_stoi += stoi
        sum_srmr += srmr
        sum_loss += loss.item()
        
        num_batches += 1

    # Calculate the mean value of each metric and loss
    mean_snr = sum_snr / num_batches
    mean_pesq = sum_pesq / num_batches
    mean_stoi = sum_stoi / num_batches
    mean_srmr = sum_srmr / num_batches
    mean_loss = sum_loss / num_batches

    print('Mean SNR:', mean_snr)
    print('Mean PESQ:', mean_pesq)
    print('Mean STOI:', mean_stoi)
    print('Mean SRMR:', mean_srmr)
    print('Mean Loss:', mean_loss)

def calculateTotalLoss(weightsOfLosses, lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss, lossDiscriminador):
    weightOfMelspecLoss1= weightsOfLosses["weightOfMelspecLoss1"]
    weightOfMelspecLoss2 = weightsOfLosses["weightOfMelspecLoss2"]
    weightOfL1Loss= weightsOfLosses["weightOfL1Loss"]
    weightOfCustomLoss =  weightsOfLosses["weightOfCustomLoss"]
    weightOfAmplitudeLoss = weightsOfLosses["weightOfAmplitudeLoss"]
    weightOfLogLoss = weightsOfLosses["weightOfLogLoss"]
    weightOfDiscriminatorLoss = weightsOfLosses["weightOfDiscriminatorLoss"]
    return lossMel1 * weightOfMelspecLoss1 + lossMel2 * weightOfMelspecLoss2 + lossAud *weightOfL1Loss + customLoss *weightOfCustomLoss + lackAmplitudeLoss * weightOfAmplitudeLoss + logLoss * weightOfLogLoss + lossDiscriminador *weightOfDiscriminatorLoss
    
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


nombreJson = "/home/afridman/wavenet/paramsRuns/runs.json"

with open(nombreJson, 'r') as file:
    runs = json.load(file)

nameFiles = runs["configRunsName"]
print(nameFiles)

for currentFile in nameFiles:
    currentFile = "/home/afridman/wavenet/paramsRuns/jsonConfigs/" + currentFile
    print(currentFile)
    with open(currentFile, 'r') as file:
        params = json.load(file)
    print("Working on " + currentFile)
    doOneRun(params)