#!/usr/bin/env python

# Copied from https://github.com/jsk-ros-pkg/jsk_recognition/blob/master/jsk_perception/node_scripts/vgg16_object_recognition.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import chainer
# from chainer import cuda
# import chainer.serializers as S
# from chainer import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
import  torchvision.transforms as transforms

from distutils.version import LooseVersion
import numpy as np
import skimage.transform

import cv_bridge
from jsk_recognition_msgs.msg import ClassificationResult
#from nin.nin import NIN
from lstm.lstm import LSTM, LSTM_torch
#from vgg16.vgg16_batch_normalization import VGG16BatchNormalization
from jsk_topic_tools import ConnectionBasedTransport  # TODO use LazyTransport
import os.path as osp
from process_gray_image import img_jet
import rospy
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import Accuracy

#from train import PreprocessedDataset
from train_torch import PreprocessedDataset


class SoundClassifier(ConnectionBasedTransport):
    """
    Classify spectrogram using neural network
    input: sensor_msgs/Image, 8UC1
    output jsk_recognition_msgs/ClassificationResult
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.gpu = rospy.get_param('~gpu', -1)
        self.train_data = rospy.get_param("~train_data", "train_data")
        self.dataset = PreprocessedDataset(transform=transforms.ToTensor(),train_data=self.train_data)
        self.target_names_ordered = self.dataset.target_classes
        self.target_names = rospy.get_param('~target_names', self.target_names_ordered)
        for i, name in enumerate(self.target_names):
            if not name.endswith('\n'):
                self.target_names[i] = name + '\n'
        self.model_name = rospy.get_param('~model_name')
        if self.model_name == 'lstm':
            self.insize = 227
            self.model = LSTM_torch(n_class=len(self.target_names))
        else:
            rospy.logerr('Unsupported ~model_name: {0}'
                         .format(self.model_name))
        ckp_path = rospy.get_param(
            '~model_file',
            osp.join(self.dataset.root, 'result_torch',
                     self.model_name, 'best_model.pt'))

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        self.model, self.optimizer = self.load_ckp(ckp_path, self.model, optimizer)

        if self.gpu == -1:
            self.device = "cpu"
        elif self.gpu == 0:
            self.device = "cuda"
            self.model.to(self.device)
            
        self.pub = self.advertise('~output', ClassificationResult,
                                  queue_size=1)
        self.pub_criteria=self.advertise("~output/criteria", Accuracy,
                                         queue_size=1)
        self.pub_input = self.advertise(
            '~debug/net_input', Image, queue_size=1)

    def load_ckp(self, checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #test_loss_min = checkpoint["valid_loss_min"]
        return model, optimizer #, checkpoint["epoch"], test_loss_min.item()

    def subscribe(self):
        sub = rospy.Subscriber(
            '~input', Image, self._recognize, callback_args=None,
            queue_size=1, buff_size=2**24)
        self.subs = [sub]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _recognize(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        mono = bridge.imgmsg_to_cv2(imgmsg)
        bgr = img_jet(mono)
        bgr = skimage.transform.resize(
            bgr, (self.insize, self.insize), preserve_range=True)
        input_msg = bridge.cv2_to_imgmsg(bgr.astype(np.uint8), encoding='bgr8')
        input_msg.header = imgmsg.header
        self.pub_input.publish(input_msg)

        # (Height, Width, Channel) -> (Channel, Height, Width)
        # ###
        #rgb = bgr.transpose((2, 0, 1))[::-1, :, :]
        rgb = bgr[:, :, ::-1]
        rgb = self.dataset.process_image(rgb)
        rgb = self.dataset.transform(rgb)
        #print(rgb)
        #x_data = np.array([rgb], dtype=np.float32)
        x_data = rgb

        #print(x_data.shape)
        x_data = x_data[np.newaxis, :, :, :]
        #print(x_data.shape)
        x_data = x_data.float()
        x_data = x_data.to(self.device)

        # swap_labels[label number in self.target_names]
        # -> label number in self.target_names_ordered
        swap_labels = [self.target_names_ordered.index(name) for name in self.target_names]
        for i in range(len(swap_labels)):
            if not (i in swap_labels):
                rospy.logerr('Wrong target_names is given by rosparam.')
                exit()
        y = self.model(x_data)
        #print(y.shape)
        proba = y.to("cpu")[0]
        #proba = cuda.to_cpu(self.model.pred.data)[0]
        #proba = (self.model.pred.data).to("cpu")[0]
        #print(proba)
        #rospy.loginfo(proba)
        proba_swapped = [proba[swap_labels[i]] for i, p in enumerate(proba)]
        #rospy.loginfo(proba_swapped)
        criteria = 0
        for i in range(len(proba_swapped)):
            criteria += (i * 1.0 / (len(proba_swapped)-1)) * proba_swapped[i]
        if criteria > 0.6:
            rospy.loginfo(criteria)
        label_swapped = np.argmax(proba_swapped)
        label_name = self.target_names[label_swapped]
        label_proba = proba_swapped[label_swapped]
        #rospy.loginfo(label_proba)
        cls_msg = ClassificationResult(
            header=imgmsg.header,
            labels=[label_swapped],
            label_names=[label_name],
            label_proba=[label_proba],
            probabilities=proba_swapped,
            classifier=self.model_name,
            target_names=self.target_names,
        )
        acc_criteria = Accuracy(
            header=imgmsg.header,
            accuracy=criteria,
            )
        self.pub.publish(cls_msg)
        self.pub_criteria.publish(acc_criteria)


if __name__ == '__main__':
    rospy.init_node('sound_classifier')
    app = SoundClassifier()
    rospy.spin()
