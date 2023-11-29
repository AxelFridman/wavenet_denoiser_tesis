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
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torchaudio.transforms as transformsaudio
import datetime
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
import time
import torch.nn.functional as F
import torchaudio.functional as Fa
import pickle
import math
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback


import librosa
import tensorflow as tf
import io
from PIL import Image
from prettytable import PrettyTable
import json

import wavenet
import postnet
import discriminator
import trainLogger

class Generator(pl.LightningModule):
    def __init__(self,
                 wavenetOfModel,postnetOfModel, discriminatorOfModel,
                    trainingHyperParameters,
                    loss_Fun , device, sample_rate,
                    howManyAudiosValidationsSave,
                    saveModelIntervalEpochs, 
                    weightsOfLosses, 
                    mel_transform1, mel_transform2 ,
                    writer, modeloNombre):

        super(Generator, self).__init__()

        self.sample_rate = sample_rate
        self.modeloNombre = modeloNombre
        self.deviceWave = device
        self.howManyAudiosValidationsSave = howManyAudiosValidationsSave
        
        self.saveModelIntervalEpochs = saveModelIntervalEpochs

        self.weightOfMelspecLoss1= weightsOfLosses["weightOfMelspecLoss1"]
        self.weightOfMelspecLoss2 = weightsOfLosses["weightOfMelspecLoss2"]
        self.weightOfL1Loss= weightsOfLosses["weightOfL1Loss"]
        self.weightOfCustomLoss =  weightsOfLosses["weightOfCustomLoss"]
        self.weightOfAmplitudeLoss = weightsOfLosses["weightOfAmplitudeLoss"]
        self.weightOfLogLoss = weightsOfLosses["weightOfLogLoss"]
        self.weightOfDiscriminatorLoss = weightsOfLosses["weightOfDiscriminatorLoss"]
        self.postnetWeightRelativeToWavenet = weightsOfLosses["postnetWeightRelativeToWavenet"]

        self.wavenet = wavenetOfModel
        self.postnet = postnetOfModel

        self.postActivateInSteps = trainingHyperParameters["postActivateInSteps"]
        self.discriminator = discriminatorOfModel
        self.discriminatorTraining = trainingHyperParameters["discriminatorTraining"]

        self.loss_fun = loss_Fun

        self.learning_rate = trainingHyperParameters["learning_rate"]

        self.learning_rate_decay_time = trainingHyperParameters["learning_rate_decay_time"]
        self.learning_rate_decay = trainingHyperParameters["learning_rate_decay"]
        self.postnetActivated = trainingHyperParameters["PostnetActivated"]

        self.loggerBoard = trainLogger.trainingLogger(deviceLogger = device, sample_rate_logger=sample_rate, writer=writer, mel_transform1=mel_transform1, mel_transform2=mel_transform2)
        self.epochNumberVal = 0


    def forward(self, x):

        output = self.wavenet(x)
        outputPost = 0
        if(self.postnetActivated):
            outputPost = self.postnet(output)

        return output, self.postnetActivated, outputPost

    def configure_optimizers(self):
        learning_rate = self.learning_rate
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.learning_rate_decay_time, gamma=self.learning_rate_decay)
        return {'optimizer': optimizer,
                'lr_scheduler': 
                    {'scheduler': scheduler,
                    'interval': 'step',  # Default is 'epoch'
                    }
                }

    def calculateTotalLoss(self,lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss, lossDiscriminador):
        return lossMel1 * self.weightOfMelspecLoss1 + lossMel2 * self.weightOfMelspecLoss2 + lossAud *self.weightOfL1Loss + customLoss *self.weightOfCustomLoss + lackAmplitudeLoss * self.weightOfAmplitudeLoss + logLoss * self.weightOfLogLoss + lossDiscriminador * self.weightOfDiscriminatorLoss
    
    def trainDiscriminator(self, y_pred, y):
        lossConFalsa = self.discriminator.step_train(y_pred, torch.tensor(0.0))
        lossConTrue = self.discriminator.step_train(y, torch.tensor(1.0))
        return (lossConFalsa, lossConTrue)

    def training_step(self, train_batch, batch_idx):
        sch = self.lr_schedulers()
        sch.step()
        #print(self.optimizers())
        #print(self.optimizers().param_groups)
        #print(self.optimizers().param_groups[0])

        self.learning_rate = self.optimizers().param_groups[0]['lr']
        batch_idx = self.epochNumberVal
        if(batch_idx>self.postActivateInSteps):
            self.postnetActivated = True


        X, y , audio_file_name, genero, pais, texto = train_batch
        self.epochNumberVal =  self.epochNumberVal + X.shape[0]

        X = X.to(self.device)
        y = y.to(self.device)

        # forward pass
        y_pred_sin_post, activado, y_pred_conPost = self.forward(X)

        if(not activado):
            y_pred = y_pred_sin_post
        else:
            y_pred = y_pred_conPost

        #Reset Lists log
        self.loggerBoard.reset_lists(X, y, y_pred, batch_idx, self.epochNumberVal)


        if(self.discriminatorTraining and ((batch_idx%2)==0)):
            self.discriminator.requires_grad_(True)

            lossConFalsa, lossConTrue = self.trainDiscriminator(y_pred, y)
            # Add discriminator loss logger
            self.loggerBoard.add_discriminator_loss()

        else:
        # compute loss
            self.discriminator.requires_grad_(False)
            if(self.discriminatorTraining):
                prediccionDiscriminador = self.discriminator.forward(y_pred)
                lossDiscriminador = 1 - prediccionDiscriminador
            else:
                lossDiscriminador = 1
            if(activado):
                lossMel1ConPost, lossMel2ConPost, customLossConPost, lossAudConPost, lackAmplitudeLossConPost, logLossConPost = self.loss_fun(y_pred_conPost, y)
                lossConPost = self.calculateTotalLoss(lossMel1ConPost, lossMel2ConPost, customLossConPost, lossAudConPost, lackAmplitudeLossConPost, logLossConPost, lossDiscriminador) 
            
            lossMel1SinPost, lossMel2SinPost, customLossSinPost, lossAudSinPost, lackAmplitudeLossSinPost, logLossSinPost = self.loss_fun(y_pred_sin_post, y)
            lossSinPost = self.calculateTotalLoss(lossMel1SinPost, lossMel2SinPost, customLossSinPost, lossAudSinPost, lackAmplitudeLossSinPost, logLossSinPost, lossDiscriminador)
            
            if(activado):
                loss = lossConPost
                lossFinal = lossConPost * self.postnetWeightRelativeToWavenet + lossSinPost
                lossMel1, lossMel2, customLoss, lossAud, lackAmplitudeLoss, logLoss = lossMel1ConPost, lossMel2ConPost, customLossConPost, lossAudConPost, lackAmplitudeLossConPost, logLossConPost

            else:
                loss = lossSinPost
                lossFinal = lossSinPost
                lossMel1, lossMel2, customLoss, lossAud, lackAmplitudeLoss, logLoss = lossMel1SinPost, lossMel2SinPost, customLossSinPost, lossAudSinPost, lackAmplitudeLossSinPost, logLossSinPost

            # add train losses log
            self.loggerBoard.add_train_losses( y_pred, y, loss, lossMel1, lossMel2, customLoss, lossAud, lackAmplitudeLoss, logLoss ,batch_idx, self.learning_rate)
            self.log('val_loss', self.loggerBoard.ValLoss.mean(), prog_bar=True)
            self.log('train_loss', loss.mean(), prog_bar=True)

            if((self.epochNumberVal % self.saveModelIntervalEpochs) == 0):
                wr = self.loggerBoard.writer
                self.loggerBoard.writer = 0
                torch.save(self, self.modeloNombre)
                self.loggerBoard.writer = wr
            return lossFinal

    def validation_step(self, val_batch, batch_idx):
        X, y , audio_file_name, genero, pais, texto= val_batch
        X = X.to(self.device)
        y = y.to(self.device)

        # forward pass
        y_pred_sin_post, activado, y_pred_conPost = self.forward(X)
        if(activado):
            y_pred = y_pred_conPost
        else:
            y_pred = y_pred_sin_post

        if(batch_idx<self.howManyAudiosValidationsSave):
            #Save audio and log val losses
            wr = self.loggerBoard.writer
            self.loggerBoard.writer = 0

            torch.save(self, self.modeloNombre)
            self.loggerBoard.writer = wr

            wave = self.modeloNombre.replace(".pth", "wave")+".pth"
            post = self.modeloNombre.replace(".pth", "post")+".pth"
            torch.save(self.wavenet, wave)
            torch.save(self.postnet, post)

            filehandler = open(self.modeloNombre+"pkl","wb")
            wtemp = self.loggerBoard
            self.loggerBoard = 0
            pickle.dump(self,filehandler)
            self.loggerBoard = wtemp
            filehandler.close()
            """
            filehandlerw = open(wave,"wb")
            pickle.dump(self.wavenet,filehandlerw)
            filehandlerw.close()            
            
            filehandlerp = open(post,"wb")
            pickle.dump(self.postnet,filehandlerp)
            filehandlerp.close()
            #file = open("Fruits.obj",'rb')
            #object_file = pickle.load(file)
            #file.close()
            """
            self.loggerBoard.save_audio_log_val_losses(X, y, y_pred, batch_idx, self.epochNumberVal)

        # compute loss
        self.discriminator.requires_grad_(False)
        prediccionDiscriminador = self.discriminator.forward(y_pred)
        lossDiscriminador = 1 - prediccionDiscriminador

        lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss = self.loss_fun(y_pred, y)
        loss = self.calculateTotalLoss(lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss, lossDiscriminador)[0][0]
        self.log('val_loss', loss.mean())

        # append log losses
        self.loggerBoard.append_log_losses_val(loss, lossMel1, lossMel2, lossAud, customLoss, lackAmplitudeLoss, logLoss)

        return loss
    def test_step(self, val_batch, batch_idx):
        X, y , audio_file_name, genero, pais, texto= val_batch
        X = X.to(self.device)
        y = y.to(self.device)

        # forward pass
        y_pred_sin_post, activado, y_pred_conPost = self.forward(X)
        if(activado):
            y_pred = y_pred_conPost
        else:
            y_pred = y_pred_sin_post
        # compute loss
        self.discriminator.requires_grad_(False)
        prediccionDiscriminador = self.discriminator.forward(y_pred)
        lossDiscriminador = 1 - prediccionDiscriminador

        lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss = self.loss_fun(y_pred, y)
        loss = self.calculateTotalLoss(lossMel1, lossMel2, customLoss, lossAud , lackAmplitudeLoss, logLoss, lossDiscriminador)
        self.log('val_loss', loss.mean())

        dic = self.loggerBoard.log_metrics(X, y, y_pred, batch_idx, self.epochNumberVal)
        
        self.log('snr', dic['snr'])
        self.log('srmr', dic['srmr'])
        self.log('pesq', dic['pesq'])
        self.log('stoi', dic['stoi'])

        self.log('lossMel1', lossMel1)
        self.log('lossMel2', lossMel2)
        self.log('customLoss', customLoss)
        self.log('lossAud', lossAud)
        self.log('lackAmplitudeLoss', lackAmplitudeLoss)
        self.log('logLoss', logLoss)


        dic['lossMel1'] = lossMel1
        dic['lossMel2'] = lossMel2
        dic['customLoss'] = customLoss
        dic['lossAud'] = lossAud
        dic['lackAmplitudeLoss'] = lackAmplitudeLoss
        dic['logLoss'] = logLoss
        return dic
     