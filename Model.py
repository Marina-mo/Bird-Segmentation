#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from Dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
import torchvision.models

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +\
                                                 target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()



def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=pretrained)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 128, 3, 1)
        self.conv_up0 = convrelu(64 + 128, 64, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 64, 32, 3, 1)

        self.conv_last = nn.Conv2d(32, 1, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        with torch.no_grad():
            layer0 = self.layer0(input)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
        

class MyModel(pl.LightningModule):
    def __init__(self, pretrained = False):
        super().__init__()

        self.model = ResNetUNet(pretrained)
        for i, l in enumerate(self.model.base_layers):
            if i!= 6 and i!=7:
                for param in l.parameters():
                    param.requires_grad = False   
        self.bce_weight = 0.95


    def forward(self, x):
        x = self.model(x)
        return x
    
    
    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch

        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)

        return {'loss': loss}
    

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=5e-4)
        
        return [optimizer]
 
        
        
def train_segmentation_model(train_dir):
    imgs_prepr = transforms.Compose([
        transforms.ToPILImage(),                               
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])
    gt_prepr = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    
    train_data = MyDataset(os.path.join(train_dir, 'images'),
                           os.path.join(train_dir, 'gt'),imgs_prepr, gt_prepr)
    dl_train = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    
    model = MyModel(pretrained = True)
    
    trainer = pl.Trainer(max_epochs=11)
    trainer.fit(model, dl_train)
    
    return model
    



    
