
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


# depth_decoder
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from .hr_layers import *



class PWSA(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(PWSA, self).__init__()

        self.Conv1x1 = Conv1x1(input_channel, input_channel)
        self.Res_block = ConvBlock(input_channel, input_channel)
        self.softmax = nn.Softmax(dim=1)
        # self.upsample = upsample()
        self.Conv1x1_out = Conv1x1(input_channel, output_channel)

    def forward(self, FD, FE):
        Sadd = (FD + FE) / 2
        Satt = self.Res_block(FE)
        Satt = self.softmax(self.Conv1x1(Satt))
        Sscaled = Sadd * Satt
        S = self.Res_block(Sscaled)
        FD_out = upsample(self.Conv1x1_out(S))

        return FD_out


class FusionDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(FusionDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.num_ch_enc = num_ch_enc  # features in encoder, [64, 18, 36, 72, 144][16,64,128,160,320]
        # self.num_ch_enc[0] = 96

        # decoder
        self.convs = OrderedDict()



        self.convs[("parallel_conv"), 2, 0] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 2, 1] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])



        self.convs[("conv1x1", 2, 2_1)] = ConvBlock1x1(self.num_ch_enc[2] + self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("attention", 2)] = fSEModule(self.num_ch_enc[2], self.num_ch_enc[3])

        self.convs[("parallel_conv"), 3, 0] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])

        self.convs[("attention", 1)] = fSEModule(self.num_ch_enc[1], self.num_ch_enc[2])

        self.convs[("up_conv"), 0] = ConvBlock(96, 48)

        self.convs[("dispconv", 0)] = Conv3x3(48, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    def FusionConv(self, conv, high_feature, low_feature):

        high_features = [upsample(high_feature)]
        high_features.append(low_feature)
        high_features = torch.cat(high_features, 1)

        return conv(high_features)

    def FusionConv_nosample(self, conv, high_feature, low_feature):
        high_features = [high_feature]
        high_features.append(low_feature)
        high_features = torch.cat(high_features, 1)

        return conv(high_features)

    def forward(self, input_features):
        self.outputs = {}

        e = updown_sample(input_features[0], 2)

        e2 = input_features[3]
        e1 = input_features[2]
        e0 = input_features[1]

        d2_1 = self.convs[("parallel_conv"), 2, 0](e0)
        d2_2 = self.convs[("parallel_conv"), 2, 1](e1)

        d23_2 = self.convs[("attention", 2)](e2, d2_2)
        d22_1 = self.FusionConv(self.convs[("conv1x1", 2, 2_1)], d2_2, d2_1)

        d3_0 = self.convs[("parallel_conv"), 3, 0](d22_1)
        d32_1 = self.convs[("attention", 1)](d23_2, d3_0)

        d = self.convs[("parallel_conv"), 3, 0](d32_1)

        d = [updown_sample(d, 2)]
        d += [e]
        d = torch.cat(d, 1)
        d = self.convs[("up_conv"), 0](d)


        self.outputs[("disp", 0)] = self.sigmoid(updown_sample(self.convs[("dispconv", 0)](d), 2))

        return self.outputs  # single-scale depth
