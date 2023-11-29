print("Importando dataset")

import torch
import torch.nn as nn
import torchaudio
import torch.fft
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

import seaborn as sns
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torchaudio.transforms as transformsaudio
import soundfile as sf

import torch.nn.functional as F
import torchaudio.functional as Fa

import math
import random

import librosa
import tensorflow as tf
import io



class AudioCleaningDataset(Dataset):
    def __init__(self, filesLocations, directoryBase,
    target_length, noiseConfigs, isTrain
    ):
        self.already_noisy_dir = ""
        self.dataframe = pd.read_csv(directoryBase+filesLocations["csvAudioFile"])
        self.audio_dir = directoryBase+filesLocations["folderAudioFiles"]
        self.noise_df = pd.read_csv(directoryBase+filesLocations["csvNoiseFile"])
        self.noise_dir = directoryBase+filesLocations["folderNoiseFiles"]
        self.reverb_dir = directoryBase+filesLocations["folderIRFiles"]
        self.target_length = target_length

        if(isTrain):
            dicRuido = noiseConfigs["trainDifficulty"]
            curriculum = noiseConfigs["learningCurriculum"]
            fixed_interval = False
        else:
            dicRuido = noiseConfigs["validationDifficulty"]
            curriculum = { "stepsTillDificultyIncrement":[],
                            "maxRuidoChange":[],
                            "SNRChange":     [],
                            "ReverbChange":  []
                        }
            fixed_interval = True

        self.ir = dicRuido["add_impulse_response"]
        self.max_noise = dicRuido["maxRuido"]
        self.snr = dicRuido["snr"]
        self.reverb_files = os.listdir(directoryBase+filesLocations["folderIRFiles"])
        self.reverb_difficulty_level = dicRuido["DificultadReverb"]

        self.min_snr_coef = dicRuido["min_snr_coef"]
        self.min_white_noise_coef = dicRuido["min_white_noise_coef"]

        self.step_number = 0
        self.steps_till_difficulty_increment = curriculum["stepsTillDificultyIncrement"]
        self.max_noise_change = curriculum["maxRuidoChange"]
        self.snr_change = curriculum["SNRChange"]
        self.reverb_change = curriculum["ReverbChange"]

        self.fixed_interval = fixed_interval
        self.use_distortions = dicRuido["useDistortions"]
        self.tanh = nn.Tanh()

    def activate_ir(self):
        self.ir=True

    def change_max_noise(self, max_noise):
        self.max_noise = max_noise

    def change_snr(self, snr):
        self.snr = snr

    def __len__(self):
        return len(self.dataframe)

    def update_difficulties_if_needed(self):
        for ind, steps_needed in enumerate(self.steps_till_difficulty_increment):
            if(steps_needed < self.step_number):
                self.max_noise = self.max_noise_change[ind]
                self.snr = self.snr_change[ind]
                self.reverb_difficulty_level = self.reverb_change[ind]
        self.step_number += 1

    def apply_distortions(self, waveform, sample_rate):
        rand_filter = random.randint(0,4)
        if rand_filter==0:
            return Fa.phaser(waveform , sample_rate=sample_rate)
        if rand_filter==1:
            return Fa.overdrive(waveform, colour = 1)
        if rand_filter==2:
            return Fa.flanger(waveform , sample_rate=sample_rate)
        if rand_filter==3:
            cutoff = random.randint(0,3000)
            return Fa.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=cutoff )
        return waveform

    def apply_reverb(self, waveform, sample_rate):
        IR_file_name = random.choice(self.reverb_files)
        IR_waveform, sample_Rate_IR = torchaudio.load(os.path.join(self.reverb_dir, IR_file_name))

        reverb_difficulty = random.uniform(0, self.reverb_difficulty_level)

        convd = reverb_difficulty * Fa.fftconvolve(waveform, IR_waveform, mode="full")
        convd = self.tanh(convd)

        return waveform + convd[:,:waveform[0].shape[0]]

    def make_target_length(self, waveform_original,waveform_dirty ):
        if(self.fixed_interval):
            waveform_original = waveform_original[:,:self.target_length]
            waveform_dirty = waveform_dirty[:,:self.target_length]
        else:
            possible_start = random.randint(0, max(0,(waveform_dirty[0].shape[0] - self.target_length)))
            waveform_original = waveform_original[:,possible_start:possible_start+self.target_length]
            waveform_dirty = waveform_dirty[:,possible_start:possible_start+self.target_length]

        padding0 = torch.zeros((1, max(self.target_length - waveform_original.size(1),0)))
        padding1 = torch.zeros((1, max(self.target_length - waveform_dirty.size(1),0)))

        waveform_original = torch.cat((waveform_original, padding0), dim=1)
        waveform_dirty = torch.cat((waveform_dirty, padding1), dim=1)
        return waveform_original, waveform_dirty
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.update_difficulties_if_needed()
        
        audio_file_name = os.path.join(self.audio_dir, self.dataframe.iloc[idx]['audio_file_name']+".wav")
        genero = self.dataframe.iloc[idx]['Gender']
        pais =  self.dataframe.iloc[idx]['Country']
        texto =  self.dataframe.iloc[idx]['Recording']
        waveform, sample_rate = torchaudio.load(audio_file_name)
        waveform_original = waveform #self.resample_audio(waveform)
        
        if(self.already_noisy_dir != ""):
            audio_file_name_noisy = os.path.join(self.already_noisy_dir, self.dataframe.iloc[idx]['audio_file_name']+".wav")
            waveform_dirty, sample_rate = torchaudio.load(audio_file_name_noisy)
            waveform_original, waveform_dirty = self.make_target_length(waveform_original, waveform_dirty)
            return waveform_dirty, waveform_original, audio_file_name, genero, pais, texto

        waveform_dirty = waveform_original.clone()

        if(self.use_distortions):
            waveform_dirty = self.apply_distortions(waveform_dirty, sample_rate)

        if(self.ir):
            waveform_dirty = self.apply_reverb(waveform_dirty, sample_rate)

        waveform_original, waveform_dirty = self.make_target_length(waveform_original, waveform_dirty)

        noise_file_name = random.choice(self.noise_df["file_name_with_directory"])

        noise_waveform, sample_rate_noise = torchaudio.load(os.path.join(self.noise_dir, noise_file_name))
        maxAbs = noise_waveform.abs().max()
        noise_waveform = (noise_waveform / maxAbs)

        while noise_waveform.size(1) < waveform_dirty.size(1):
            noise_waveform = torch.cat((noise_waveform, noise_waveform), dim=1)
        
        #noise_waveform = noise_waveform[:,:waveform_dirty.size(1)]
        possible_start = random.randint(0, max(0,(noise_waveform[0].shape[0] - self.target_length)))
        noise_waveform = noise_waveform[:,possible_start:possible_start+self.target_length]

        current_snr = random.uniform(self.snr * self.min_snr_coef, self.snr)
        waveform_dirty = waveform_dirty + (noise_waveform * current_snr)

        white_noise = random.uniform(self.max_noise * self.min_white_noise_coef, self.max_noise)
        waveform_dirty = waveform_dirty + torch.randn_like(waveform_original) * white_noise

        maxAbs = waveform_dirty.abs().max()
        waveform_dirty = waveform_dirty / maxAbs
        return waveform_dirty, waveform_original,1,1,1,1#