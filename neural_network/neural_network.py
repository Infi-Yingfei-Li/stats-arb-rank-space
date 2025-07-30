import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional

import matplotlib.pyplot as plt

class feed_forward_neural_network(nn.Module):
    def __init__(self, input_feature_num, drop_out_rate):
        super(feed_forward_neural_network, self).__init__()
        self.drop_out_rate = drop_out_rate

        self.linear1 = nn.Linear(input_feature_num, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 8)
        self.linear4 = nn.Linear(8, 1)
        self.ReLu1 = nn.PReLU(); self.ReLu2 = nn.PReLU(); self.ReLu3 = nn.PReLU()
        self.dropout1 = nn.Dropout(self.drop_out_rate); self.dropout2 = nn.Dropout(self.drop_out_rate); self.dropout3 = nn.Dropout(self.drop_out_rate)

    def forward(self, x):
        x = (x-torch.mean(x, dim=-1, keepdim=True))/(torch.max(torch.std(x, dim=-1, keepdim=True), 1e-5*torch.ones(x.shape).to(x.device)))
        x = self.linear1(x); x = self.ReLu1(x); x = self.dropout1(x)
        x = self.linear2(x); x = self.ReLu2(x); x = self.dropout2(x)
        x = self.linear3(x); x = self.ReLu3(x); x = self.dropout3(x)
        x = self.linear4(x)
        return x

class convolution_neural_network(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, drop_out_rate, ReLU_type="PReLU"):
        super(convolution_neural_network, self).__init__()
        self.input_channels = input_channels; self.output_channels = output_channels; self.kernel_size = kernel_size; self.drop_out_rate = drop_out_rate

        self.padding_num = (int((kernel_size-1)/2), int((kernel_size-1)/2)) if kernel_size % 2 == 1 else (int((kernel_size)/2), int(kernel_size/2-1))
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size)
        if ReLU_type == "PReLU":
            self.ReLU1 = nn.PReLU(); self.ReLU2 = nn.PReLU()
        if ReLU_type == "ReLU":
            self.ReLU1 = nn.ReLU(); self.ReLU2 = nn.ReLU()

        self.normalization1 = nn.InstanceNorm1d(input_channels)
        self.normalization2 = nn.InstanceNorm1d(output_channels)

    def forward(self, x):
        '''
        params:
            x (torch.tensor): (batch_size, input_channels, seq_length)
        returns:
            y (torch.tensor): (batch_size, output_channels, seq_length)
        '''
        x = self.normalization1(x)
        y = torch.nn.functional.pad(x, self.padding_num, mode="constant", value=0)
        y = self.conv1(y)
        y = self.ReLU1(y)
        y = self.normalization2(y)
        y = torch.nn.functional.pad(y, self.padding_num, mode="constant", value=0)
        y = self.conv2(y)
        y = self.ReLU2(y)
        y = y + x.repeat(1, int(self.output_channels/self.input_channels), 1)
        return y

class transformer(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, head_num, drop_out_rate):
        super(transformer, self).__init__()
        self.input_channels = input_channels; self.output_channels = output_channels; self.hidden_channels = hidden_channels
        self.head_num = head_num; self.drop_out_rate = drop_out_rate
        self.d_k = self.hidden_channels//self.head_num
        self.W_Q = nn.Sequential(nn.Linear(self.input_channels, self.hidden_channels), nn.Dropout(self.drop_out_rate))
        self.W_K = nn.Sequential(nn.Linear(self.input_channels, self.hidden_channels), nn.Dropout(self.drop_out_rate))
        self.W_V = nn.Sequential(nn.Linear(self.input_channels, self.hidden_channels), nn.Dropout(self.drop_out_rate))
        self.W_O = nn.Sequential(nn.Linear(self.hidden_channels, 2*self.hidden_channels), nn.Dropout(self.drop_out_rate),
                                 nn.ReLU(), nn.Linear(2*self.hidden_channels, self.output_channels), nn.Dropout(self.drop_out_rate))
        self.add_norm_1 = nn.LayerNorm(self.input_channels)
        self.add_norm_2 = nn.LayerNorm(self.output_channels)

    def forward(self, x):
        '''
        params:
            x (torch.tensor): (batch_size, input_channels, seq_length)
        returns:
            output (torch.tensor): (batch_size, output_channels, seq_length)
        '''
        x = torch.transpose(x, 1, 2)
        Q = self.split_head(self.W_Q(x)); K = self.split_head(self.W_K(x)); V = self.split_head(self.W_V(x))
        attention_output = self.combine_head(self.attention_scale_dot_product(Q, K, V))
        attention_output = self.add_norm_1(attention_output+x)
        output = self.W_O(attention_output)
        output = self.add_norm_2(output+attention_output)
        return output

    def split_head(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.head_num, self.d_k).transpose(1,2)

    def combine_head(self, x):
        batch_size, head_num, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, head_num*d_k)

    def attention_scale_dot_product(self, Q, K, V):
        attention_score = torch.matmul(Q, K.transpose(-2, -1))/torch.sqrt(torch.tensor(self.d_k))
        attention_prob = torch.softmax(attention_score, dim=-1)
        attention_output = torch.matmul(attention_prob, V)
        return attention_output

class CNN_transformer(nn.Module):
    def __init__(self, CNN_input_channels, CNN_output_channels, CNN_kernel_size, CNN_drop_out_rate, \
                 transformer_input_channels, transformer_hidden_channels, transformer_output_channels, transformer_head_num, transformer_drop_out_rate):
        super(CNN_transformer, self).__init__()
        if CNN_output_channels != transformer_input_channels:
            raise ValueError("CNN_output_channels must be equal to transformer_input_channels.")
        self.CNN = convolution_neural_network(CNN_input_channels, CNN_output_channels, CNN_kernel_size, CNN_drop_out_rate)
        #self.transformer = transformer(transformer_input_channels, transformer_hidden_channels, transformer_output_channels, transformer_head_num, transformer_drop_out_rate)
        self.transformer = nn.TransformerEncoderLayer(d_model=transformer_input_channels, nhead=transformer_head_num, dim_feedforward=transformer_hidden_channels, dropout=transformer_drop_out_rate, batch_first=True)
        self.linear = nn.Linear(transformer_output_channels, 1)
        #self.FFN = nn.Sequential(nn.Linear(transformer_output_channels, 8), nn.PReLU(), nn.Dropout(transformer_drop_out_rate),
        #                         nn.Linear(8,4), nn.PReLU(), nn.Dropout(transformer_drop_out_rate), nn.Linear(4,1))
    
    def forward(self, x):
        '''
        params:
            x (torch.tensor): (batch_size, CNN_input_channels, seq_length)
        '''
        x = self.CNN(x) # (batch_size, CNN_output_channels, seq_length)
        x = x.permute(0, 2, 1) # (batch_size, seq_length, CNN_output_channels)
        x = self.transformer(x) # (batch_size, seq_length, transformer_output_channels)
        x = x[:, -1, :] # (batch_size, transformer_output_channels)
        x = self.linear(x) # (batch_size, 1) 
        #x = self.FFN(x)
        return x

