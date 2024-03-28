# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_scatter
import math
from tasks.semantic.modules.SalsaNextCompletionSingleDecoder import SalsaNextCompletionSingleDecoder

def get_gaussian_kernel(device_arg, kernel_size=3, sigma=0.2, channels=1):
    # create an x,y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size, device=device_arg.device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance))*torch.exp(-torch.sum((xy_grid-mean)**2., dim=-1)/(2*variance))

    # make sure the sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

    return gaussian_kernel


class mlp(nn.Module):
    def __init__(self, input_dim, hidden_ratio, output_dim):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim*hidden_ratio))
        self.norm = nn.BatchNorm1d(int(input_dim*hidden_ratio))
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(int(input_dim * hidden_ratio), output_dim)
        self.part = 50000

    def forward(self, feats):
        num_non_zero_locations, channel_size, neighbours = feats.shape
        feats = feats.permute(0,2,1).contiguous()
        feats = feats.reshape(-1, channel_size)
        if feats.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = feats.shape[0] // self.part
            part_linear_out = [self.fc1(
                feats[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts+1)
            ]
            feats = torch.cat(part_linear_out, dim=0)
        else:
            feats = self.fc1(feats)
        feats = self.act(self.norm(feats))
        if feats.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = feats.shape[0] // self.part
            part_lienar_out = [self.fc2(
                feats[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts+1)
            ]
            feats = torch.cat(part_lienar_out, dim=0)
        else:
            feats = self.fc2(feats)
        feats = feats.reshape(num_non_zero_locations, neighbours, -1)
        feats = feats.permute(0,2,1).contiguous()
        return feats


class vfe(nn.Module):
    def __init__(self, input_dim, hidden_ratio, output_dim, kernel_size):
        super(vfe, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim * hidden_ratio))
        self.norm = nn.BatchNorm1d(int(input_dim * hidden_ratio))
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(int(input_dim * hidden_ratio), output_dim)
        self.kernel_size = kernel_size
        self.maxpool = nn.MaxPool1d(self.kernel_size, stride=1)
        self.avgpool = nn.AvgPool1d(self.kernel_size, stride=1)
        self.out = output_dim
        self.part = 50000

    def forward(self, feats):
        num_non_zero_locations, channel_size, neighbours = feats.shape
        feats = feats.permute(0, 2, 1).contiguous()
        feats = feats.reshape(-1, channel_size)
        if feats.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = feats.shape[0] // self.part
            part_lienar_out = [self.fc1(
                feats[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)
            ]
            feats = torch.cat(part_lienar_out, dim=0)
        else:
            feats = self.fc1(feats)
        feats = self.act(self.norm(feats))
        if feats.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = feats.shape[0] // self.part
            part_lienar_out = [self.fc2(
                feats[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)
            ]
            feats = torch.cat(part_lienar_out, dim=0)
        else:
            feats = self.fc2(feats)
        feats = feats.reshape(num_non_zero_locations, neighbours, -1)
        feats = feats.permute(0, 2, 1).contiguous()
        _, channel_size, _ = feats.shape
        feats = feats.reshape(-1, neighbours)
        try:
            max = self.maxpool(feats)
            avg = self.avgpool(feats)
        except:
            max, _ = torch.max(feats, dim=1, keepdim=True)
            avg = torch.mean(feats, dim=1, keepdim=True)

        feats = feats.reshape(num_non_zero_locations, channel_size, -1)
        max = max.reshape(num_non_zero_locations, channel_size, -1)
        avg = avg.reshape(num_non_zero_locations, channel_size, -1)
        feats = torch.cat((feats, max.repeat(1, 1, neighbours), avg.repeat(1, 1, neighbours)), dim=-2)
        return feats

class continuous_convolution_network(nn.Module):
    def __init__(self, search_kernel, k_neighbour, dilation=1):
        super(continuous_convolution_network, self).__init__()
        self.kernel_size = search_kernel
        self.k_neighbour = k_neighbour
        self.coords = 5
        self.vfe_hidden_ratio = 2
        self.vfe_output = int(self.coords * 4)
        self.mlp_input = 3 * self.vfe_output
        self.mlp_output = 5
        self.mlp_hidden_ratio = 2
        self.vfe = vfe(self.coords, self.vfe_hidden_ratio, self.vfe_output, self.k_neighbour)
        self.mlp = mlp(self.mlp_input, self.mlp_hidden_ratio, self.mlp_output)
        self.dilation = dilation

        self.uf = nn.Unfold(kernel_size=self.kernel_size, padding=(self.kernel_size-1)*self.dilation//2, dilation=self.dilation)
        self.post_linear = nn.Sequential(
            nn.Linear(self.coords * self.k_neighbour, self.coords),
            nn.BatchNorm1d(self.coords),
            nn.ReLU(),
            nn.Linear(self.coords, self.coords)
        )

    def forward(self, input_coords):
        b, c, h, w = input_coords.shape

        # obtain relative coordinates (in terms of range and remission)
        uf_coords = self.uf(input_coords).reshape(b, c, self.kernel_size**2, h, w)
        mask = torch.sum(uf_coords[:,0], dim=1) > 0 # a mask to determine if neighbourhoods have any projected points
        uf_rel_coords = (uf_coords - uf_coords[:,:,[self.kernel_size ** 2//2],...]).abs() # get relative distance
        uf_coords = uf_coords.permute(0,3,4,1,2)[mask] # shape: [num_entries, 5, k]
        uf_rel_coords = uf_rel_coords.permute(0,3,4,1,2)[mask] # shape: [num_entries, 5, k]

        # find closest neighbours
        uf_distance = uf_rel_coords[:,0] + uf_rel_coords[:,-1] + 0.01 # 0.01 is added so that the gaussian kernel can still apply to zero-location
        inv_gauss_k = (1 - get_gaussian_kernel(uf_distance, self.kernel_size, 10, 1)).view(1, -1)
        uf_distance = uf_distance * inv_gauss_k

        _, uf_distance_idx = uf_distance.topk(
            self.k_neighbour, dim=1, largest=False, sorted=True
        )

        uf_coords = torch.gather(
            input=uf_coords, dim=2, index=uf_distance_idx.unsqueeze(1).repeat(1,5,1)
        ) # shape: [num_non_zero_locations, 5, num_neighbours]
        # to do: ensure the middle entry is excluded from input
        # compute spacial attention values
        att = self.vfe(uf_coords)
        att = self.mlp(att)
        att = F.softmax(att, dim=2)
        # apply attention weights
        uf_coords *= att
        uf_coords = uf_coords.view(uf_coords.shape[0], -1)
        uf_coords = self.post_linear(uf_coords)
        return_coords = torch.zeros((b,h,w,c), device=input_coords.device)
        return_coords[mask] = uf_coords
        return_coords = return_coords.permute(0,3,1,2)
        return return_coords

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class SalsaNextCompletionSingleDecoderStageTwoPerceptionLoss(nn.Module):
    def __init__(self, nclasses):
        super(SalsaNextCompletionSingleDecoderStageTwoPerceptionLoss, self).__init__()
        self.nclasses = nclasses
        self.completion_network = SalsaNextCompletionSingleDecoder(nclasses)

        for param in self.completion_network.parameters():
            # if config.option == 'frozen':
            # param.requires_grad = False
            # elif config.option == 'flexible':
            param.requires_grad = False
        self.downCntx_s = ResContextBlock(5, 32)
        self.downCntx2_s = ResContextBlock(32, 32)
        self.downCntx3_s = ResContextBlock(32, 32)

        self.resBlock1_s = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2_s = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3_s = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4_s = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5_s = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1_s = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2_s = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3_s = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4_s = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.logits_s = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

        # self.completor = continuous_convolution_network(9, 9)
    def forward(self, x, proj_mask):
        completed_scene = self.completion_network(x, proj_mask)
        downCntx_s = self.downCntx_s(completed_scene)
        downCntx_s = self.downCntx2_s(downCntx_s)
        downCntx_s = self.downCntx3_s(downCntx_s)

        down0c_s, down0b_s = self.resBlock1_s(downCntx_s)
        down1c_s, down1b_s = self.resBlock2_s(down0c_s)
        down2c_s, down2b_s = self.resBlock3_s(down1c_s)
        down3c_s, down3b_s = self.resBlock4_s(down2c_s)
        down5c_s = self.resBlock5_s(down3c_s)

        up4e_s = self.upBlock1_s(down5c_s,down3b_s)
        up3e_s = self.upBlock2_s(up4e_s, down2b_s)
        up2e_s = self.upBlock3_s(up3e_s, down1b_s)
        up1e_s = self.upBlock4_s(up2e_s, down0b_s)
        logits_s = self.logits_s(up1e_s)

        logits_s = logits_s
        logits_s = F.softmax(logits_s, dim=1)
        return logits_s, up1e_s