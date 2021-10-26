#!/usr/bin/env python

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
import  torchvision.transforms as transforms

import matplotlib
import numpy as np
from os import makedirs
import os.path as osp
from PIL import Image as Image_
import rospkg
import rospy

from nin.nin import NIN
from lstm.lstm import LSTM_torch

import shutil

matplotlib.use("Agg")

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, path=None, random=True, transform= None, train_data="train_data"):
        rospack = rospkg.RosPack()
        self.root = osp.join(rospack.get_path(
            "sound_classification"), train_data)
        if path is not None:
            self.base = torchvision.datasets.ImageFolder(
                root=osp.join(self.root, path))
        self.random = random
        # how many classes to be classified
        self.n_class = 0
        self.target_classes = []
        self.transform = transform
        with open(osp.join(self.root, 'n_class.txt'), mode='r') as f:
            for row in f:
                self.n_class += 1
                self.target_classes.append(row)
        # Load mean image of dataset
        mean_img_path = osp.join(rospack.get_path('sound_classification'),
                                 train_data, 'dataset_torch', 'mean_of_dataset.png')
        # mean = np.array(Image_.open(mean_img_path), np.float32).transpose(
        #     (2, 0, 1))  # (height, width, channel) -> (channel ,height, width), rgb
        #self.mean = mean.astype(chainer.get_dtype())

        mean = np.array(Image_.open(mean_img_path), np.float32)  # (height, width, channel)
        self.mean = mean
        

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, label = self.base.samples[i]
        #a = np.array(Image_.open(self.base.samples[i][0]))
        #print(a.shape) #(227, 227, 3)
        image = np.array(Image_.open(self.base.samples[i][0]), dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        image = self.process_image(image)

        image = self.transform(image)
        return image, label

    def process_image(self, image):
        ret = image - self.mean
        ret *= (1.0 / 255.0)
        return ret

def load_model(model_name, n_class):
    archs = {
        #'nin': NIN,
        #'vgg16': VGG16BatchNormalization
        "lstm": LSTM_torch
    }
    model = archs[model_name](n_class=n_class)
    return model

def save_ckp(state, best_model_path):
    torch.save(state, best_model_path)
        
def main():
    rospack = rospkg.RosPack()

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument("-t", "--train_data", type=str, default="train_data")
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-m', '--model', type=str,
                        choices=["lstm"], default="lstm",
                        help='Neural network model to use dataset')
    args = parser.parse_args()


    trans1 = transforms.ToTensor()
    if args.gpu == -1:
        device = "cpu"
    elif args.gpu == 0:
        device = "cuda"
        
    batchsize = 32
    # Path to training image-label list file
    #train_labels = osp.join(rospack.get_path('sound_classification'),
    #                        args.train_data, 'dataset', 'train_images.txt')
    # Path to validation image-label list file
    #val_labels = osp.join(rospack.get_path('sound_classification'),
    #                      args.train_data, 'dataset', 'test_images.txt')

    # Initialize the model to train
    print('Device: {}'.format(device))
    print('Model: {}'.format(args.model))
    #print('Dtype: {}'.format(chainer.config.dtype))
    print('Minibatch-size: {}'.format(batchsize))
    print('epoch: {}'.format(args.epoch))
    print('')

    train = PreprocessedDataset("dataset_torch", transform = trans1, train_data=args.train_data)
    #print(train.__getitem__(0)[0].size())
    #print(train.__getitem__(0)[1])
    #train.__getitem__(0)
    val = PreprocessedDataset("dataset_torch_val", transform = trans1, train_data=args.train_data)

    model = load_model(args.model, train.n_class)
    model = model.to(device)

    trainloader = torch.utils.data.DataLoader(train, batch_size=batchsize, shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(val, batch_size=batchsize, shuffle=False, num_workers=1)

    dataloader_dict = {
        "train": trainloader,
        "test": testloader
    }

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    out = osp.join(rospack.get_path("sound_classification"),
                   args.train_data, "result_torch", args.model)
    best_model_path = osp.join(out, "best_model.pt")
    if not osp.exists(out):
        makedirs(out)

    #batch_iterator = iter(dataloader_dict["train"])
    #inputs, labels = next(batch_iterator)
    #print(inputs.size())
    #print(labels)

    num_epochs = 30
    test_loss_min = 10
    for epoch in range(num_epochs):
        print("-----")

        train_loss = 0.0
        test_loss = 0.0

        train_corrects  = 0.0
        test_corrects = 0.0
        
        ##train phase
        model.train()
        for inputs, labels in dataloader_dict["train"]:
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                outputs = outputs.to("cpu")
                #calculate loss
                loss = criterion(outputs, labels)
                #predict labels
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_corrects += torch.sum(preds == labels.data)

        #loss and accuracy for each epoch
        train_loss = train_loss / len(dataloader_dict["train"].dataset)
        train_acc = train_corrects.double() / len(dataloader_dict["train"].dataset)

        print("train Loss: {:.4f} Acc: {:.4f}".format(train_loss, train_acc))

        ###test phase
        model.eval()
        for inputs, labels in dataloader_dict["test"]:
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.to("cpu")
            #calculate loss
            loss = criterion(outputs, labels)
            #predict labels
            _, preds = torch.max(outputs, 1)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

        #loss and accuracy for each epoch
        test_loss = test_loss / len(dataloader_dict["test"].dataset)
        test_acc = test_corrects.double() / len(dataloader_dict["test"].dataset)

        print("test Loss: {:.4f} Acc: {:.4f}".format(test_loss, test_acc))

        checkpoint = {
            "epoch": epoch+1,
            "valid_loss_min": test_loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        
        if test_loss <= test_loss_min:
            save_ckp(checkpoint, best_model_path)
            test_loss_min = test_loss

if __name__ == "__main__":
    main()
