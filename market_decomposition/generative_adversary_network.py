import os, sys, copy, h5py, datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


stochastic_discount_factor_network_config = {"input_size": 2, "rnn_hidden_size": 2, "ffn_hidden_size": 2, "ffn_output_size": 1}

class stochastic_discount_factor_network(nn.Module):
    def __init__(self, stochastic_discount_factor_network_config):
        super(stochastic_discount_factor_network, self).__init__()
        self.config = stochastic_discount_factor_network_config
        self.lstm = nn.LSTM(self.config["input_size"], self.config["rnn_hidden_size"], batch_first=True, dropout=0.05)
        self.ffn = nn.Sequential(nn.Linear(self.config["rnn_hidden_size"], self.config["ffn_hidden_size"]), nn.ReLU(), nn.Dropout(p=0.05),
                                    nn.Linear(self.config["ffn_hidden_size"], self.config["ffn_output_size"]), nn.ReLU(), nn.Dropout(p=0.05))

    def forward(self, x):
        '''
        params:
            x: (batch_size, seq_len, input_size)
        '''
        c_0 , h_0 = (torch.zeros(1, x.size(0), self.config["rnn_hidden_size"]), torch.zeros(1, x.size(0), self.config["rnn_hidden_size"]))
        output, (final_hidden_state, final_cell_state) = self.lstm(x, (c_0, h_0))
        result = self.ffn(final_hidden_state)
        return result


