import pickle
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .rnn_nn import *
from .base_classifier import *


class RNN_Classifier(Base_Classifier):
    
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()

    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    def forward(self, x):
    ############################################################################
    #  TODO: Perform the forward pass                                          #
    ############################################################################   


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return x


class LSTM_Classifier(Base_Classifier):

    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        self.lstm = nn.LSTM(input_size, hidden_size, 3)
        self.l1 = nn.Linear(hidden_size, classes)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################


    def forward(self, x):

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################
        # print(x.shape)    
        x, _ = self.lstm(x)
        # print(x.shape)
        x = x[-1,:,:]
        # x = self.l2(x)
        # x = torch.relu(x)
        # x = nn.Dropout(p=0.2)
        x = self.l1(x)
        

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return x
