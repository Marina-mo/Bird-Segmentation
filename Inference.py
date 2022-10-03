#!/usr/bin/env python
# coding: utf-8

from glob import glob
from os.path import join, splitext
from os.path import split, dirname, abspath
from sys import argv
import sys
import getopt
from skimage.io import imread, imsave
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from Model import MyModel, train_segmentation_model


def run_on_image(image_path):
    code_dir = dirname(abspath(__file__))
    model = MyModel(False)
    model.load_state_dict(torch.load(join(code_dir, 'segmentation_model.pth'), map_location=torch.device('cpu')))
    segm = (predict(model, image_path) > 0.5).astype('uint8') * 255
    head, tail = split(image_path)
    name, ext = splitext(tail)
    imsave(join(head, name + '_pred' + '.png'), segm)

def run_on_test(train = False):
    code_dir = dirname(abspath(__file__))

    if train:
        model = train_segmentation_model(join(code_dir, '00_test_val_input/train'))
    else:
        model = MyModel(False)
        model.load_state_dict(torch.load(join(code_dir, 'segmentation_model.pth'), map_location=torch.device('cpu')))

    img_filenames = glob(join('00_test_val_input', 'test/images/**/*.jpg'))
    for filename in img_filenames:
        segm = (predict(model, filename) > 0.5).astype('uint8') * 255
        head, tail = split(filename)
        name, ext = splitext(tail)
        imsave(join(head, name + '_pred' + '.png'), segm)


def predict(model, img_path):
    imgs_prepr = transforms.Compose([
    transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    print(img_path)

    raw_img = cv2.imread(img_path)

    img = imgs_prepr(raw_img)
    model.eval()
    with torch.no_grad():
        pred = model.forward(img[None,:,:])
        sig_pred = torch.sigmoid(pred)
        
    up = nn.Upsample(size =(raw_img.shape[0], raw_img.shape[1]), mode = 'bilinear')
        
    return up(sig_pred)[0,0,:,:].numpy()


if __name__ == '__main__':
    print(argv)
    try:
        opts, args = getopt.getopt(argv[2:],"ht:i:",["help","train=","infile="])
    except getopt.GetoptError:
        print('Inference.py mode -t <train> -i <infile>')
        sys.exit(2)
    
    mode = argv[1]
    train = False
    infile = ''

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('Inference.py mode -t <train> -i <infile>')
            sys.exit()
        elif opt in ("-t", "--train"):
            train = arg
        elif opt in ("-i", "--infile"):
            infile = arg

    if mode == 'run_on_test':
        run_on_test(train)
    if mode == 'run_on_image':
        print(infile)
        run_on_image(infile)
 
    
