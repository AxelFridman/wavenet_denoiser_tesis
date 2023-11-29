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


import librosa
import tensorflow as tf
import io
from PIL import Image
from prettytable import PrettyTable
import json

from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import SpeechReverberationModulationEnergyRatio
from torchmetrics.audio import SignalNoiseRatio

class trainingLogger():
    def __init__(self, deviceLogger, sample_rate_logger, writer, mel_transform1 , mel_transform2):
        self.sample_rate_logger = sample_rate_logger
        self.writer = writer
        self.deviceLogger = deviceLogger
        self.mel_transform1 = mel_transform1
        self.mel_transform2 = mel_transform2

        self.srmr = SpeechReverberationModulationEnergyRatio(sample_rate_logger).to(deviceLogger)
        self.wb_pesq = PerceptualEvaluationSpeechQuality(sample_rate_logger, 'wb').to(deviceLogger)
        self.stoi = ShortTimeObjectiveIntelligibility(sample_rate_logger, False).to(deviceLogger)
        self.snr = SignalNoiseRatio(sample_rate_logger).to(deviceLogger)

        self.ValLoss = 0
        self.valMel1Loss = 0
        self.valMel2Loss = 0
        self.valL1Loss = 0
        self.valPropLoss = 0
        self.vallogLoss = 0
        self.valamplitudeLoss = 0
        self.TrainLoss = 0



        self.PESQValLoss =0
        self.STOIValLoss = 0
        self.SRMRValLoss = 0
        self.FWSSNRValLoss = 0
        self.SNRValLoss = 0

        self.PESQValLosses =[]
        self.STOIValLosses = []
        self.SRMRValLosses = []
        self.FWSSNRValLosses = []
        self.SNRValLosses = []

        self.ValLosses = []
        self.valMel1Losses=[]
        self.valMel2Losses=[]
        self.valL1Losses=[]
        self.valPropLosses=[]
        self.vallogLosses=[]
        self.valamplitudeLosses=[]


    def generateWaveformPlot(self, predictedAudio, realAudio, noisyAudio, batch_idx, epoch):
        predictedAudio=predictedAudio.numpy().flatten()
        realAudio=realAudio.numpy().flatten()
        noisyAudio=noisyAudio.numpy().flatten()

        plt.figure(figsize=(10, 6))  # Set the figure size

        # Plot original audio
        sns.lineplot(x=range(0, len(realAudio)), y=realAudio, label="Original")

        # Plot noisy audio
        sns.lineplot(x=range(0, len(noisyAudio)), y=noisyAudio, label="Ruidoso")

        # Plot predicted audio
        sns.lineplot(x=range(0, len(predictedAudio)), y=predictedAudio, label="Predicho")

        plt.title(f'Waveform_{batch_idx}_{epoch}')

        # Save the figure as an image
        figure_path = f'waveform_{batch_idx}_{epoch}.png'
        plt.savefig(figure_path)
        plt.close()  # Close the figure to free up memory

        # Open the saved image
        with open(figure_path, 'rb') as img_file:
            image_data = img_file.read()

        # Convert the image data to a PyTorch tensor
        image_tensor = torch.from_numpy(plt.imread(figure_path)).permute(2, 0, 1)

        # Add the image to TensorBoard
        self.writer.add_image(f'waveform_{batch_idx}_{epoch}', image_tensor, global_step=epoch)

        # Remove the saved image file after it's logged to TensorBoard (optional)
        os.remove(figure_path)

    def prepareSpectogram2(self, melsp, name, batch_idx, epoch):
        mel_spectrogram_db = transformsaudio.AmplitudeToDB()(melsp).cpu()
        # Display the mel spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram_db.detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(format="%+2.0f dB")
        plt.title(f'melspec{batch_idx}_{epoch}_{name}')
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency Bin")
        # Save the figure as an image
        figure_path = f'melspec_{batch_idx}_{epoch}.png'
        plt.savefig(figure_path)
        plt.close()
        # Open the saved image
        with open(figure_path, 'rb') as img_file:
            image_data = img_file.read()

        # Convert the image data to a PyTorch tensor
        image_tensor = torch.from_numpy(plt.imread(figure_path)).permute(2, 0, 1)

        # Add the image to TensorBoard
        self.writer.add_image(f'melspec{batch_idx}_{epoch}_{name}', image_tensor, global_step=epoch)

        # Remove the saved image file after it's logged to TensorBoard (optional)
        os.remove(figure_path)

    def generatePlots(self, predictedAudio, realAudio, noisyAudio, batch_idx, epoch):
        self.generateWaveformPlot(predictedAudio, realAudio, noisyAudio, batch_idx, epoch)
        mel1true_y = self.mel_transform1(realAudio.to(self.deviceLogger))
        mel1y_pred = self.mel_transform1(predictedAudio.to(self.deviceLogger))
        mel1sucio = self.mel_transform1(noisyAudio.to(self.deviceLogger))
        self.prepareSpectogram2(mel1y_pred,"PredichoPrecicionFrec", batch_idx, epoch)
        self.prepareSpectogram2(mel1sucio[0],"RuidosoPrecicionFrec", batch_idx, epoch)
        self.prepareSpectogram2(mel1true_y[0],"OriginalPrecicionFrec", batch_idx, epoch)

    def add_discriminator_loss(self):
        self.writer.add_scalars("Losses de discriminador", {'Loss cuando es falso audio':lossConFalsa,
                'Loss cuando es verdadero':lossConTrue
                }, batch_idx)
    
    def add_train_losses(self, y_pred, y, loss, lossMel1, lossMel2, customLoss, lossAud, lackAmplitudeLoss, logLoss ,batch_idx, learning_rate):
        trainLosssrmr = self.srmr(y_pred)
        trainLossstoi = self.stoi(y_pred, y)
        try:
            trainLosspesq = self.wb_pesq(y_pred, y)
        except:
            trainLosspesq = self.PESQValLoss

        trainLosssnr = self.snr(y_pred, y)
        self.writer.add_scalars("Learning rate", {'LR':learning_rate
                        }, batch_idx)
        
        self.writer.add_scalars("STOI", {'train':trainLossstoi,
                        'validation':self.STOIValLoss
                        }, batch_idx)
        
        self.writer.add_scalars("SRMR", {'train':trainLosssrmr,
                        'validation':self.SRMRValLoss
                        }, batch_idx)
        self.writer.add_scalars("SNR", {'train':trainLosssnr,
                        'validation':self.SNRValLoss
                        }, batch_idx)

        self.writer.add_scalars("PESQ", {'train':trainLosspesq,
                                        'validation':self.PESQValLoss
                                        }, batch_idx)


        self.writer.add_scalars("Loss L1", {'train':lossAud,
                                'validation':self.valL1Loss
                                }, batch_idx)
        self.writer.add_scalars("Loss prop", {'train':customLoss,
                        'validation':self.valPropLoss
                        }, batch_idx)
        self.writer.add_scalars("Loss melspect1", {'train':lossMel1,
                        'validation':self.valMel1Loss
                        }, batch_idx)
        self.writer.add_scalars("Loss melspect2", {'train':lossMel2,
                        'validation':self.valMel2Loss
                        }, batch_idx)


        self.writer.add_scalars("Loss amplitude diference", {'train':lackAmplitudeLoss,
                'validation':lackAmplitudeLoss
                }, batch_idx)
        self.writer.add_scalars("Loss logarithmic", {'train':logLoss,
                'validation':logLoss
                }, batch_idx)

        self.writer.add_scalars("Loss total", {'train':loss.mean(),
                                'validation':self.ValLoss.mean()
                                }, batch_idx)


    def save_audio_log_val_losses(self, X, y, y_pred, batch_idx, epochNumberVal):
        audio_clip = X[0].cpu().numpy()
        self.writer.add_audio(f'audio_clip_{batch_idx}_Sucio', audio_clip, global_step=batch_idx, sample_rate=self.sample_rate_logger)
        audio_clip = y[0].cpu().numpy()
        self.writer.add_audio(f'audio_clip_{batch_idx}_Original', audio_clip, global_step=batch_idx, sample_rate=self.sample_rate_logger)
        audio_clip = y_pred[0].cpu()#.numpy()
        self.writer.add_audio(f'audio_clip_{batch_idx}_{epochNumberVal}', audio_clip, global_step=batch_idx,sample_rate=self.sample_rate_logger)
        self.generatePlots(audio_clip[0], y[0].cpu(), X[0].cpu(), batch_idx, epochNumberVal)
        
        self.log_metrics(X, y, y_pred, batch_idx, epochNumberVal)

    def log_metrics(self, X, y, y_pred, batch_idx, epochNumberVal):
        try:
            pesq = self.wb_pesq(y_pred, y).cpu()
            self.PESQValLosses.append(pesq) # self.wb_pesq(y_pred, y)
        except:
            pesq = 0
        stoi = self.stoi(y_pred, y).cpu()
        self.STOIValLosses.append(stoi)
        srmr = self.srmr(y_pred).cpu()
        self.SRMRValLosses.append(srmr)
        self.FWSSNRValLosses.append(0)
        snr = self.snr(y_pred, y).cpu()
        self.SNRValLosses.append(snr)
        return {'snr':snr, 'srmr':srmr, 'stoi':stoi, 'pesq':pesq}


    def reset_lists(self, X, y, y_pred, batch_idx, epochNumberVal):
        if(len(self.ValLosses)>0):
            audio_clip = X[0].cpu().detach().numpy()
            self.writer.add_audio(f'audio_clip_train_{batch_idx}_{epochNumberVal}_noise', audio_clip, global_step=batch_idx,sample_rate=self.sample_rate_logger)
            audio_clip = y[0].cpu().detach().numpy()
            self.writer.add_audio(f'audio_clip_train_{batch_idx}_{epochNumberVal}_clean', audio_clip, global_step=batch_idx,sample_rate=self.sample_rate_logger)
            audio_clip = y_pred[0].cpu().detach()
            self.writer.add_audio(f'audio_clip_train_{batch_idx}_{epochNumberVal}_predicted', audio_clip, global_step=batch_idx,sample_rate=self.sample_rate_logger)
            
            self.generatePlots(audio_clip[0], y[0].cpu(), X[0].cpu(), batch_idx, epochNumberVal)
            
            self.ValLoss = np.array(self.ValLosses).mean()
            self.valMel1Loss = np.array(self.valMel1Losses).mean()
            self.valMel2Loss = np.array(self.valMel2Losses).mean()
            self.valL1Loss = np.array(self.valL1Losses).mean()
            self.valPropLoss = np.array(self.valPropLosses).mean()

            self.ValLosses = []
            self.valMel1Losses=[]
            self.valMel2Losses=[]
            self.valL1Losses=[]
            self.valPropLosses=[]
            self.vallogLoss=[]
            self.valamplitudeLoss=[]

            self.PESQValLoss =np.array(self.PESQValLosses).mean()
            self.STOIValLoss = np.array(self.STOIValLosses).mean()
            self.SRMRValLoss = np.array(self.SRMRValLosses).mean()
            self.FWSSNRValLoss = np.array(self.FWSSNRValLosses).mean()
            self.SNRValLoss = np.array(self.SNRValLosses).mean()

            self.PESQValLosses =[]
            self.STOIValLosses = []
            self.SRMRValLosses = []
            self.FWSSNRValLosses = []
            self.SNRValLosses = []

    def append_log_losses_val(self, loss, lossMel1, lossMel2, lossAud, customLoss, lackAmplitudeLoss, logLoss):
        self.ValLosses.append(loss.mean().item())
        self.valMel1Losses.append(lossMel1.cpu())
        self.valMel2Losses.append(lossMel2.cpu())
        self.valL1Losses.append(lossAud.cpu())
        self.valPropLosses.append(customLoss.cpu())

        self.valamplitudeLoss = lackAmplitudeLoss
        self.vallogLoss= logLoss

