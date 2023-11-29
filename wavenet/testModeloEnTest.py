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
import datasetParaTest
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
    wavenetOfModel = wavenet.WaveNet(wavenetParams , device)
    postnetOfModel = postnet.PostNetSimple(postnetParams, device)
    discriminatorOfModel = discriminator.Discriminator(device)

    melspec1 = generateMelTransform(sample_rate, lossesConfig["melspec1Config"], device)
    melspec2 = generateMelTransform(sample_rate, lossesConfig["melspec2Config"], device)
    
    nombreParaGuardarLog = nameOfRun#.split("/")
    #nombreParaGuardarLog = nombreParaGuardarLog[-2] + nombreParaGuardarLog[-1] 
    writer = SummaryWriter(log_dir=fileLocations["saveTensorboardLocation"] + "/"+nombreParaGuardarLog , comment=nameOfRun)
    #%load_ext tensorboard
    loss_Fun = losses.CombinedLoss(melspec1, melspec2, device, weightsOfLosses["max_weight_short_pitch"], lossesConfig["epsilon"], lossesConfig["min_db_precision"])

    modeloNombre = nameOfRun+'.pth'
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

    file = open("/home/afridman/CorridaFinalTInf.pthpkl",'rb')
    model = pickle.load(file)
    file.close()
   
    #model.load_state_dict(torch.load("/home/afridman/pretrained_weights.pth"))



    print("Importadas todas las librerias y modelo")
    
    
    fullvaldataset = datasetParaTest.AudioCleaningDataset(fileLocations["FullValLocations"], 
                                                   fileLocations["directoryBase"],
                                    target_length=inputSize, noiseConfigs=noiseConfigs, isTrain = False )


    fullValDataloader = DataLoader(fullvaldataset, batch_size=batch_size, shuffle=False, num_workers=16)


    srmrf = SpeechReverberationModulationEnergyRatio(sample_rate).to(device)
    wb_pesqf = PerceptualEvaluationSpeechQuality(sample_rate, 'wb').to(device)
    stoif = ShortTimeObjectiveIntelligibility(sample_rate, False).to(device)
    snrf = SignalNoiseRatio(sample_rate).to(device)

    # Iterate over the entire validation set
    #model.to("cpu")
    generos = ["m","f"]
    paises = ["cl", "pr", "co", "ar", "pe", "ve"]
    tiposCorrida = ["wave", "limpio", "ruido"]
    list_snr, list_pesq, list_stoi, list_srmr, list_loss, generosvistos, paisesvistos, textos = [],[],[],[],[],[],[],[]
    audioidvistos = {}
    numbatches = []
    numuestra = []
    sizes = []
    tardanzas = []
    tiposdecorridalista = []
    reverbNivel = []
    ruidoNivel = []
    ruidoDeTodoslosBatches = noiseConfigs["validationDifficulty"]["snr"]
    reverbDeTodoslosBatches = noiseConfigs["validationDifficulty"]["DificultadReverb"]

    model.eval()
    with torch.no_grad():
        for muestra in range(0,1):
            print(muestra)
            num_batches = 0

            for val_batch in fullValDataloader:
                X, y , audio_file_name, genero, pais, texto= val_batch
                size = (X.shape[2])
                print(size)
                if(size<300000):
                    try:
                        genero = genero[0]
                        pais = pais[0]
                        texto = texto[0]
                        audio_file_name = audio_file_name[0]
                        X = X.to(device)
                        y = y.to(device)
                        ini = time.time()
                        retorno = model(X)
                        fin = time.time()
                        tardanza = fin - ini

                        predichoSoloWave = retorno[0]
                        predichoConPost = retorno[2]

                        sucio = X
                        corridas = [predichoSoloWave,  y, sucio]
                        for i in range(0, len(corridas)):
                            actual = corridas[i]
                            tipo = tiposCorrida[i]

                            lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss = loss_Fun(actual, y)
                            lossDiscriminador = 0
                            loss = calculateTotalLoss(weightsOfLosses,lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss, lossDiscriminador).to("cpu")

                            snr = snrf(actual,y).to("cpu")#compute_snr(y, y_pred)
                            pesq = wb_pesqf(actual,y).to("cpu")#compute_pesq(y, y_pred)
                            stoi = stoif(actual,y).to("cpu")#compute_stoi(y, y_pred)
                            srmr = srmrf(actual).to("cpu")#compute_srmr(y, y_pred)

                            list_snr.append(float(snr))
                            list_pesq.append(float(pesq))
                            list_stoi.append(float(stoi))
                            list_srmr.append(float(srmr))
                            list_loss.append(float(loss.item()))
                            generosvistos.append(genero)
                            paisesvistos.append(pais)
                            textos.append(texto)
                            numbatches.append(num_batches)
                            numuestra.append(muestra)
                            sizes.append(size)
                            tardanzas.append(tardanza)
                            ruidoNivel.append(ruidoDeTodoslosBatches)
                            reverbNivel.append(reverbDeTodoslosBatches)
                            noiseConfigs["validationDifficulty"]["snr"]
                            tiposdecorridalista.append(tipo)
                            print(num_batches)
                        num_batches += 1
                    except:
                        pass
                    if(num_batches%10==0):
                        datamue = {'snr': list_snr, 'pesq': list_pesq, 
                        'stoi': list_stoi,'srmr': list_srmr,
                        'loss': list_loss, 'textos': textos, 
                        'genero': generosvistos,'pais': paisesvistos, 
                        "muestra": numuestra , "batch":numbatches,
                        "ruido":ruidoNivel,"reverb":reverbNivel,
                        "size": sizes, "tiempoInferencia":tardanzas,
                        "tipo":tiposdecorridalista}
                        df = pd.DataFrame(datamue)
                        df.to_csv("/home/afridman/varianzaResultados"+str(ruidoDeTodoslosBatches)+".csv")
            
        if(num_batches >= 10):
            datamue = {'snr': list_snr, 'pesq': list_pesq, 
            'stoi': list_stoi,'srmr': list_srmr,
            'loss': list_loss, 'textos': textos, 
            'genero': generosvistos,'pais': paisesvistos, 
            "muestra": numuestra , "batch":numbatches,
            "ruido":ruidoNivel,"reverb":reverbNivel,
            "size": sizes, "tiempoInferencia":tardanzas,
              "tipo":tiposdecorridalista}
            df = pd.DataFrame(datamue)
            df.to_csv("/home/afridman/varianzaResultados"+str(ruidoDeTodoslosBatches)+".csv")
            
    #print(audioidvistos)

    """
    with torch.no_grad():
        num_batches = 0
        for val_batch in fullValDataloader:
            X, y , audio_file_name, genero, pais, texto= val_batch
            genero = genero[0]
            pais = pais[0]
            texto = texto[0]
            X = X.to(device)
            y = y.to(device)
            # TODO DESCOMENTAR SI SE ESTA MIRANDO MODELO
            X = model(X)[0]# en vez de 0 poner 2 si tiene post
            print(X.shape)
            
            lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss = loss_Fun(X, y)
            lossDiscriminador = 0
            loss = calculateTotalLoss(weightsOfLosses,lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss, lossDiscriminador).to("cpu")
            
            # compute metrics
            snr = snrf(X,y).to("cpu")#compute_snr(y, y_pred)
            pesq = wb_pesqf(X,y).to("cpu")#compute_pesq(y, y_pred)
            stoi = stoif(X,y).to("cpu")#compute_stoi(y, y_pred)
            srmr = srmrf(X).to("cpu")#compute_srmr(y, y_pred)

            # Add metrics and loss to the corresponding sum

            
            list_snr.append(float(snr))
            list_pesq.append(float(pesq))
            list_stoi.append(float(stoi))
            list_srmr.append(float(srmr))
            list_loss.append(float(loss.item()))
            generosvistos.append(genero)
            paisesvistos.append(pais)
            textos.append(texto)
            num_batches += 1
    """

    data = {'snr': list_snr, 'pesq': list_pesq, 
            'stoi': list_stoi,'srmr': list_srmr,
             'loss': list_loss, 'textos': textos, 
            'genero': generosvistos,'pais': paisesvistos}#.to("cpu")
    df = pd.DataFrame(data)
    #print(df)
    #df.to_csv("/home/afridman/resultadosTest2.csv")

    

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


nombreJson = "/home/afridman/wavenet/paramsRuns/tests.json"

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