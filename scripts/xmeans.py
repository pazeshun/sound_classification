#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import cluster, preprocessing

import argparse
from os import makedirs
import os.path as osp
from PIL import Image as Image_
import rospkg
import rospy

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from nin.nin import NIN
#matplotlib.use("Agg")
import sys

def process_image(image, mean):
    ret = image - mean
    ret *= (1.0 / 255.0)
    return ret

def main():
    rospack = rospkg.RosPack()
    root = osp.join(rospack.get_path(
        "sound_classification"), "610_data")

    mean_img_path = osp.join(rospack.get_path("sound_classification"),
                             "610_data", "dataset", "mean_of_dataset.png")
    mean = np.array(Image_.open(mean_img_path), np.float32).transpose(
        (2, 0, 1))

    img_dir = osp.join(rospack.get_path("sound_classification"),
                        "kimura_data", "dataset")

    images = np.empty((0, 3))
    target_class = []
    # with open(osp.join(root, "n_class.txt"), mode="r") as f:
    #     for row in f:
    #         target_class.append(row)

    target_class = ["strong", "weak", "no_sound"]

    model = NIN(len(target_class), xmeans=True)

    for j in target_class:
        for i in range(200):
            #print(j)
            j = j.rstrip("\n")
            img_path = osp.join(img_dir, "train_" + str(j) +  "_" + str(j) +  "_{0:05d}_000.png".format(i))
            if not osp.exists(img_path):
                continue
            print(img_path)
            image = np.array(Image_.open(img_path), np.float32).transpose(
                (2, 0, 1))
            preprocessed_image = process_image(image, mean)

            preprocessed_image = preprocessed_image.reshape(1, 3, 227, 227)
            
            y = model(preprocessed_image)
            y = y.data
            #print(y.shape)

            #sys.exit()
            #a = preprocessed_image.reshape(1, -1)
            y = y.reshape(1, -1)
            images = np.vstack((images, y))
            #print(preprocessed_image.shape)
            #print(images.shape)

    # sc = preprocessing.StandardScaler()
    # sc.fit(images)
    # X_norm = sc.transform(images)

    X_norm = images

    #X-means
    # xm_c = kmeans_plusplus_initializer(X_norm, 2).initialize()
    # xm_i = xmeans(data=X_norm, initial_centers=xm_c, kmax=10, ccore=True)
    # xm_i.process()

    # z_xm = np.ones(X_norm.shape[0])
    # for k in range(len(xm_i._xmeans__clusters)):
    #     z_xm[xm_i._xmeans__clusters[k]] = k+1

    # print(z_xm)

    #Spectral Clustring
    km = cluster.SpectralClustering(n_clusters=3)
    z_km = km.fit(X_norm)

    print(z_km.labels_)


def test():
    df_wine_all = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine = df_wine_all[[0, 10, 13]]
    df_wine.columns = [u'class', u'color', u'proline']

    X = df_wine[["color", "proline"]]
    sc = preprocessing.StandardScaler()
    sc.fit(X)
    X_norm = sc.transform(X)
    print(X_norm)

    x = X_norm[:,0]
    y = X_norm[:,1]
    z=df_wine["class"]
    plt.figure(figsize=(10,10))
    plt.subplot(4, 1, 1)
    plt.scatter(x,y, c=z)
    
    xm_c = kmeans_plusplus_initializer(X_norm, 2).initialize()
    xm_i = xmeans(data=X_norm, initial_centers=xm_c, kmax=20, ccore=True)
    xm_i.process()

    z_xm = np.ones(X_norm.shape[0])
    for k in range(len(xm_i._xmeans__clusters)):
        z_xm[xm_i._xmeans__clusters[k]] = k+1

    plt.subplot(4,1,2)
    plt.scatter(x,y,c=z_xm)
    centers = np.array(xm_i._xmeans__centers)
    plt.scatter(centers[:,0],centers[:,1],s=250, marker='*',c='red')
    plt.show()

if __name__ == "__main__":
    main()
    #test()
