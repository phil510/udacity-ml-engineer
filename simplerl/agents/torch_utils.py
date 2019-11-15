import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def calc_n_step_returns(reward, value_estimate, terminal, 
                        gamma = 1.0, n_steps = 1, seq_model = False):
    assert (reward.dim() == 2), 'TODO'
    assert (terminal.shape == reward.shape), 'TODO'
    
    assert (value_estimate.dim() == 3), 'TODO'
    assert (value_estimate.shape[0] == reward.shape[0]), 'TODO'
    assert (value_estimate.shape[1] == reward.shape[1]), 'TODO'
    
    assert (n_steps <= reward.shape[1])
    
    seq_len = reward.shape[1]
    output_len = seq_len if seq_model else 1
    
    returns = []
    for i in reversed(range(output_len)):
        R = value_estimate[:, min(i + n_steps, seq_len) - 1]
        for j in reversed(range(n_steps)):
            if (i + j) >= seq_len:
                continue

            R = (reward[:, i + j].view(-1, 1) + 
                 gamma * R * (1 - terminal[:, i + j].view(-1, 1)))
        
        returns.append(R.unsqueeze(1))

    returns = returns[::-1]    
    returns = torch.cat(returns, dim = 1)
    
    return returns

def to_tensors(*args, dtype = torch.float32, device = None):
    tensors = tuple()
    for item in args:
        tensor = torch.as_tensor(item, dtype = dtype, device = device)
        tensors += (tensor, )
        
    return tensors

def qr_huber_loss(difference, k = 1.0):
    loss = torch.where(difference.abs() < k, 
                       0.5 * difference.pow(2), 
                       k * (difference.abs() - 0.5 * k))
                       
    return loss

class DenseResidualBlock(nn.Module):
    def __init__(self, input_size, layers, batch_norm = True):
        super().__init__()
        self._batch_norm = batch_norm
        
        dense_layers = []
        if self._batch_norm:
            bn_layers = []
        
        for _ in range(layers):
            dense_layers.append(nn.Linear(input_size, input_size, 
                                          bias = (not batch_norm)))
            if self._batch_norm:
                bn_layers.append(nn.BatchNorm1d(input_size))
            
        self.dense_layers = nn.ModuleList(dense_layers)
        if self._batch_norm:
            self.bn_layers = nn.ModuleList(bn_layers)
        
    def forward(self, x):
        residual = x
        
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            if self._batch_norm:
                x = self.bn_layers[i](x)
            x = F.relu(x) # apply activation after bn like original paper
        
        x += residual
        
        return x
        
class LearnedScaler(nn.Module):
    def __init__(self, num_features, eps = 1e-6):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('n_obs', torch.tensor(0, dtype = torch.long))
        
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.n_obs.zero_()
        
    def forward(self, x):
        if self.training:
            for obs in x.contiguous().view(-1, self.num_features):
                self.n_obs += 1
                
                if self.n_obs > 1:
                    self.running_var = ((float(self.n_obs) - 2) 
                                        / (float(self.n_obs) - 1) 
                                        * self.running_var 
                                        + (obs - self.running_mean).pow(2) 
                                        / self.n_obs)

                self.running_mean += (obs - self.running_mean) / self.n_obs
                    
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

# from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# https://arxiv.org/pdf/1803.01271.pdf author's implementation
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, 
                 stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, 
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, 
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, 
                                 self.dropout1, self.conv2, self.chomp2, 
                                 self.relu2, self.dropout2)
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1) if 
                           n_inputs != n_outputs else None)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, 
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
        
class TCNBlock(nn.Module):
    def __init__(self, input_size, channels, 
                 kernel_size = 2, dropout = 0.2, seq_last = True):       
        super().__init__()
        
        self._seq_last = seq_last
        
        layers = []
        n_layers = len(channels)
        for i in range(n_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else channels[i - 1]
            out_channels = channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, 
                                     kernel_size = kernel_size, 
                                     stride = 1, 
                                     dilation = dilation,
                                     padding = (kernel_size - 1) * dilation, 
                                     dropout = dropout)]

        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, mask = None):
        assert (x.dim() == 3), 'TODO'
        if not self._seq_last:
            x = x.transpose(2, 1)
            
        if mask is not None:
            assert (mask.dim() == 2), 'TODO'
            assert (x.shape[0] == mask.shape[0]), 'TODO'
            mask = mask.unsqueeze(1)
            assert (mask.shape[2] == x.shape[2]), 'TODO'
            
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask
                
        return x