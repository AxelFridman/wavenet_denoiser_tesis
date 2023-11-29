print("Importando wavenet")
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
class DilatedCausalConv1d(torch.nn.Module):
    def __init__(self, channels, dilation, device):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(channels, channels,
                                    kernel_size=3, stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=dilation,  # Fixed for WaveNet dilation
                                    bias=True)  
        self.conv = self.conv.to(device)

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output


class CausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(CausalConv1d, self).__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=3, stride=1, padding=1,
                                    bias=True)  # Fixed for WaveNet but not sure
        self.conv = self.conv.to(device)

    def forward(self, x):
        output = self.conv(x)

        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation, device, use_batch_norm, use_dropout):

        super(ResidualBlock, self).__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation, device=device)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

        self.conv_skip = self.conv_skip.to(device)
        self.conv_res = self.conv_res.to(device)

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.batchNormParams = nn.BatchNorm1d(128)
        self.dropOut = nn.Dropout(use_dropout)

    def forward(self, x, skip_size):

        output = self.dilated(x)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        if(self.use_batch_norm):
            output = self.batchNormParams(output)
        output = self.dropOut(output)

        # Residual network
        output = self.conv_res(gated)
        
        if(self.use_batch_norm):
            output = self.batchNormParams(output)
        output = self.dropOut(output)

        output += x

        # Skip connection
        skip = self.conv_skip(gated)
        
        if(self.use_batch_norm):
            output = self.batchNormParams(output)
        output = self.dropOut(output)

        skip = skip[:, :, -skip_size:]

        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels,device,  use_batch_norm, use_dropout):

        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.res_blocks = self.stack_res_block(res_channels, skip_channels, device)

    def _residual_block(self, res_channels, skip_channels, dilation, device):
        block = ResidualBlock(res_channels, skip_channels, dilation, device, self.use_batch_norm, self.use_dropout)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)

        block.to(device)

        return block

    def build_dilations(self):
        dilations = []

        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)

        return dilations

    def stack_res_block(self, res_channels, skip_channels, device):

        res_blocks = []
        dilations = self.build_dilations()

        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, dilation, device)
            res_blocks.append(block)

        return res_blocks

    def forward(self, x, skip_size):

        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            # output is the next input
            output, skip = res_block(output, skip_size)

            skip_connections.append(skip)


        return torch.stack(skip_connections)


class DensNet(torch.nn.Module):
    def __init__(self, channels, device, use_batch_norm, use_dropout):

        super(DensNet, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.batchNormParams = nn.BatchNorm1d(1)
        self.dropOut = nn.Dropout(use_dropout)

        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.tan = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        self.conv1 = self.conv1.to(device)
        self.conv2 = self.conv2.to(device)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        if(self.use_batch_norm):
            output = self.batchNormParams(output)
        output = self.dropOut(output)
        output = self.conv2(output)
        #output = self.tan(output)
        return output




class WaveNet(pl.LightningModule):
    def __init__(self, wavenetParams , device):


        super(WaveNet, self).__init__()

        layer_size = wavenetParams["layer_size"]
        stack_size = wavenetParams["stack_size"]
        in_channels = wavenetParams["in_channels"]
        res_channels = wavenetParams["res_channels"]
        use_batch_norm_Wave = wavenetParams["use_batch_norm_Wave"]
        use_dropout_Wave = wavenetParams["use_dropout_Wave"]

        self.causal = CausalConv1d(in_channels, res_channels, device)

        self.res_stack = ResidualStack(layer_size, stack_size, res_channels, in_channels, device=device,  use_batch_norm=use_batch_norm_Wave, use_dropout=use_dropout_Wave )

        self.densnet = DensNet(in_channels, device, use_batch_norm_Wave, use_dropout_Wave)

        self.use_batch_norm_Wave = use_batch_norm_Wave
        self.use_dropout_Wave = use_dropout_Wave


    def change_loss_function(self, loss_fun):
        self.loss_fun = loss_fun

    def give_state(self):
        # Save additional attributes along with the model state_dict
        extra_state = {
            "use_batch_norm_Wave": self.use_batch_norm_Wave,
            "use_dropout_Wave": self.use_dropout_Wave,

            "causal": self.causal ,
            "res_stack": self.res_stack,
            "densnet": self.densnet 
        }

        # Combine model state_dict and extra state
        state_dict = {
            "model": self.state_dict(),
            "extra_state": extra_state,
        }

        # Save the combined state
        return state_dict
    def forward(self, x):

        output = x#.transpose(1, 2)


        output = self.causal(output)

        #print(x.shape)
        skip_connections = self.res_stack(output, x.shape[2]) #TODO SACAR

        output = torch.sum(skip_connections, dim=0)

        output = self.densnet(output)

        return output
 
