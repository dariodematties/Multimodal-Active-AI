# This is a logistic regresion module used to classify the image representations returned by our model

import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
        def __init__(self, input_size, num_classes):
                super(LogisticRegression, self).__init__() 
                self.linear = nn.Linear(input_size, num_classes) 

        def forward(self, x):
                out = self.linear(x)
                return out
