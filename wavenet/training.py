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
    if torch.cuda.is_available():
        print("CUDA esta disponible")
    else:
        print("CUDA no esta disponible (estas sin GPU)")

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
    modeloNombre = fileLocations['directoryBase']+"/"+ nameOfRun+'.pth'
    print(modeloNombre)
    loss_Fun = losses.CombinedLoss(melspec1, melspec2, device, weightsOfLosses["max_weight_short_pitch"], lossesConfig["epsilon"], lossesConfig["min_db_precision"])
    
    wavenetOfModel = wavenet.WaveNet(wavenetParams , device)
    postnetOfModel = postnet.PostNetSimple(postnetParams, device)
    discriminatorOfModel = discriminator.Discriminator(device)
    model = generator.Generator(
                    wavenetOfModel = wavenetOfModel,
                    postnetOfModel = postnetOfModel,
                    discriminatorOfModel = discriminatorOfModel,
                    trainingHyperParameters = trainingHyperParameters,
                    
                    loss_Fun =loss_Fun, device=device, sample_rate =sample_rate,
                    howManyAudiosValidationsSave= savingAndLogging["howManyAudiosValidationsSave"],
                    saveModelIntervalEpochs= savingAndLogging["saveModelIntervalEpochs"], 

                    weightsOfLosses = weightsOfLosses, 
                    mel_transform1=melspec1, mel_transform2=melspec2 ,
                    writer=writer, modeloNombre = modeloNombre)

    if(savingAndLogging["useSavedModel"]):
        model = torch.load(savingAndLogging["nameOfModelSaved"], map_location=torch.device('cuda'))
    model = model.to(device)
    model.postnetActivated = trainingHyperParameters["PostnetActivated"]

    if(trainingHyperParameters["discriminatorRestart"]):
        model.discriminator = discriminator.Discriminator(device)

    if(trainingHyperParameters["wavenetParamsFrozen"]):
        model.causal.requires_grad_(False)
        model.res_stack.requires_grad_(False)
        model.densnet.requires_grad_(False)

    preTraindataset = dataset.AudioCleaningDataset(fileLocations["preTrainLocations"], 
                                                   fileLocations["directoryBase"],
                                    target_length=inputSize, noiseConfigs=noiseConfigs, isTrain = True 
                                    )

    fineTuneTraindataset = dataset.AudioCleaningDataset(fileLocations["FineTuneLocations"], 
                                                   fileLocations["directoryBase"],
                                    target_length=inputSize, noiseConfigs=noiseConfigs, isTrain = False )

    valdataset = dataset.AudioCleaningDataset(fileLocations["MiniValLocations"], 
                                                   fileLocations["directoryBase"],
                                    target_length=inputSize, noiseConfigs=noiseConfigs, isTrain = False )
    
    fullvaldataset = dataset.AudioCleaningDataset(fileLocations["FullValLocations"], 
                                                   fileLocations["directoryBase"],
                                    target_length=inputSize, noiseConfigs=noiseConfigs, isTrain = False )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=savingAndLogging["min_delta"], patience=savingAndLogging["patience"], verbose=True, mode="min")


    calls = [early_stop_callback]

    if device=="cuda":
        acc = 'gpu'
    else:
        acc = 'cpu'
    preTrainTrainer = pl.Trainer(default_root_dir=fileLocations["saveTensorboardLocation"],
                        accelerator=acc, devices=1,
                            min_epochs=trainingHyperParameters["preTrainEpochs"],
                            max_epochs= trainingHyperParameters["preTrainEpochs"],
                            log_every_n_steps=trainingHyperParameters["log_every_n_steps"],
                            callbacks=calls,
                        accumulate_grad_batches=trainingHyperParameters["accumulate_grad_batches"]
                            ,val_check_interval= savingAndLogging["val_check_interval"]
                            ,reload_dataloaders_every_n_epochs=1
    )
    finetuneTrainer = pl.Trainer(default_root_dir=fileLocations["saveTensorboardLocation"],
                            accelerator=acc, devices=1,
                              min_epochs=trainingHyperParameters["fineTuneEpochs"],
                            max_epochs= trainingHyperParameters["fineTuneEpochs"],
                            log_every_n_steps=trainingHyperParameters["log_every_n_steps"],
                            callbacks=calls,
                        accumulate_grad_batches=trainingHyperParameters["accumulate_grad_batches"]
                            ,val_check_interval= savingAndLogging["val_check_interval"]
                            ,reload_dataloaders_every_n_epochs=1
    )
    pretraindataloader = DataLoader(preTraindataset, batch_size=batch_size, shuffle=True, num_workers=16)
    finetunedataloader = DataLoader(fineTuneTraindataset, batch_size=batch_size, shuffle=True, num_workers=16)

    valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False, num_workers=16)


    preTrainTrainer.fit(model=model, train_dataloaders=pretraindataloader
                , val_dataloaders=valdataloader
                )
    torch.save(model.state_dict(), "pretrained_weights.pth")

    # Load the weights before fine-tuning
    model.load_state_dict(torch.load("pretrained_weights.pth"))

    finetuneTrainer.fit(model=model, train_dataloaders=finetunedataloader
                , val_dataloaders=valdataloader
                )
    
    fullValDataloader = DataLoader(fullvaldataset, batch_size=batch_size, shuffle=False, num_workers=16)
    results = finetuneTrainer.test(dataloaders=fullValDataloader)
    results[0]["name"] = nameOfRun
    saveResults(results[0], fileLocations["directoryBase"])

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

print("New row added to the CSV file successfully!")

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