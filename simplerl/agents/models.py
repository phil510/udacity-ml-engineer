import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_utils import TCNBlock, LearnedScaler

class DDPGNet(nn.Module):
    def __init__(self, input_size, action_size, hidden_layers, 
                 action_scale = 1.0):
        super().__init__()
        
        self.action_scale = action_scale
        
        critic_layers = []
        policy_layers = []
        
        input = input_size
        for i, units in enumerate(hidden_layers):
            output = units
            policy_layers.append(nn.Linear(input, output))
            if i == 0:
                input += action_size
            critic_layers.append(nn.Linear(input, output))
            input = output
            
        self.critic_output = nn.Linear(input, 1)
        self.policy_output = nn.Linear(input, action_size)
            
        self.critic_layers = nn.ModuleList(critic_layers)
        self.policy_layers = nn.ModuleList(policy_layers)
        
        self.critic_params = (list(self.critic_layers.parameters()) 
                              + list(self.critic_output.parameters()))
        self.policy_params = (list(self.policy_layers.parameters()) 
                              + list(self.policy_output.parameters()))
        
    def forward(self, obs):
        x = obs
        for layer in self.policy_layers:
            x = F.relu(layer(x))
            
        x = torch.tanh(self.policy_output(x))
        x = x * self.action_scale
        
        return x
        
    def action(self, obs):
        return self.forward(obs)
        
    def critic_value(self, obs, action):
        x = torch.cat([obs, action], dim = -1)
        for layer in self.critic_layers:
            x = F.relu(layer(x))
            
        x = self.critic_output(x)
        
        return x
        
class TD3Net(nn.Module):
    def __init__(self, input_size, action_size, hidden_layers, 
                 action_scale = 1.0):
        super().__init__()
        
        self.action_scale = action_scale
        
        critic_layers_1 = []
        critic_layers_2 = []
        policy_layers = []
        
        input = input_size
        for i, units in enumerate(hidden_layers):
            output = units
            policy_layers.append(nn.Linear(input, output))
            if i == 0:
                input += action_size
            critic_layers_1.append(nn.Linear(input, output))
            critic_layers_2.append(nn.Linear(input, output))
            input = output
            
        self.critic_output_1 = nn.Linear(input, 1)
        self.critic_output_2 = nn.Linear(input, 1)
        self.policy_output = nn.Linear(input, action_size)
            
        self.critic_layers_1 = nn.ModuleList(critic_layers_1)
        self.critic_layers_2 = nn.ModuleList(critic_layers_2)
        self.policy_layers = nn.ModuleList(policy_layers)
        
        self.critic_params_1 = (list(self.critic_layers_1.parameters()) 
                                + list(self.critic_output_1.parameters()))
        self.critic_params_2 = (list(self.critic_layers_2.parameters()) 
                                + list(self.critic_output_2.parameters()))
                              
        self.policy_params = (list(self.policy_layers.parameters()) 
                              + list(self.policy_output.parameters()))
        
    def forward(self, obs):
        x = obs
        for layer in self.policy_layers:
            x = F.relu(layer(x))
            
        x = torch.tanh(self.policy_output(x))
        x = x * self.action_scale
        
        return x
        
    def action(self, obs):
        return self.forward(obs)
        
    def critic_value(self, obs, action):
        x1 = self.critic_value_1(obs, action)
        x2 = self.critic_value_2(obs, action)
        
        return x1, x2
        
    def critic_value_1(self, obs, action):
        x = torch.cat([obs, action], dim = -1)
        for layer in self.critic_layers_1:
            x = F.relu(layer(x))
            
        x = self.critic_output_1(x)
        
        return x
        
    def critic_value_2(self, obs, action):
        x = torch.cat([obs, action], dim = -1)
        for layer in self.critic_layers_2:
            x = F.relu(layer(x))
            
        x = self.critic_output_2(x)
        
        return x
        
class DDPGTemporalConvNet(nn.Module):
    def __init__(self, input_size, 
                 action_size, 
                 tcn_layers,
                 hidden_layers,
                 tcn_kernel_size = 2,
                 action_scale = 1.0,
                 tcn_dropout = 0.2,
                 hidden_dropout = 0.25,
                 feature_scaler = True):
        super().__init__()
        
        self.action_scale = action_scale
        
        self.scale = feature_scaler
        if feature_scaler:
            self.feature_scaler = LearnedScaler(input_size)
        
        self.policy_tcn = TCNBlock(input_size, tcn_layers, 
                                   kernel_size = tcn_kernel_size, 
                                   dropout = tcn_dropout, 
                                   seq_last = False)
                            
        self.critic_tcn = TCNBlock(input_size, tcn_layers, 
                                   kernel_size = tcn_kernel_size, 
                                   dropout = tcn_dropout, 
                                   seq_last = False)
                
        policy_layers = []
        critic_layers = []
        
        input = tcn_layers[-1]
        for i, units in enumerate(hidden_layers):
            output = units
            policy_layers.append(nn.Linear(input, output))
            if i == 0:
                input += action_size
            critic_layers.append(nn.Linear(input, output))
            input = output
            
            policy_layers.append(nn.ReLU())
            critic_layers.append(nn.ReLU())
            
            policy_layers.append(nn.Dropout(p = hidden_dropout))
            critic_layers.append(nn.Dropout(p = hidden_dropout))
            
        self.policy_layers = nn.Sequential(*policy_layers)
        self.critic_layers = nn.Sequential(*critic_layers)
        
        self.policy_output = nn.Linear(input, action_size)
        self.critic_output = nn.Linear(input, 1)
        
        self.critic_params = (list(self.critic_layers.parameters()) 
                              + list(self.critic_output.parameters())
                              + list(self.critic_tcn.parameters()))
        self.policy_params = (list(self.policy_layers.parameters()) 
                              + list(self.policy_output.parameters())
                              + list(self.policy_tcn.parameters()))
                              
    def forward(self, obs, mask = None):
        x = obs
        if self.scale:
            x = self.feature_scaler(x)
            
        x = self.policy_tcn(x, mask = mask)
        x = x[:, :, -1] # get the last output in the sequence
        
        x = self.policy_layers(x)
        
        x = torch.tanh(self.policy_output(x))
        x = x * self.action_scale
        
        return x
        
    def action(self, obs, mask = None):
        return self.forward(obs, mask = mask)
        
    def critic_value(self, obs, action, mask = None):
        x = obs
        if self.scale:
            x = self.feature_scaler(x)
        
        x = self.critic_tcn(x, mask = mask)
        x = x[:, :, -1] # get the last output in the sequence
        
        x = torch.cat([x, action], dim = -1)
        x = self.critic_layers(x)
        
        x = self.critic_output(x)
        
        return x
        
class TD3TemporalConvNet(nn.Module):
    def __init__(self, input_size, 
                 action_size, 
                 tcn_layers,
                 hidden_layers,
                 tcn_kernel_size = 2,
                 action_scale = 1.0,
                 tcn_dropout = 0.2,
                 hidden_dropout = 0.25,
                 feature_scaler = True):
        super().__init__()
        
        self.action_scale = action_scale
        
        self.scale = feature_scaler
        if feature_scaler:
            self.feature_scaler = LearnedScaler(input_size)
        
        self.policy_tcn = TCNBlock(input_size, tcn_layers, 
                                   kernel_size = tcn_kernel_size, 
                                   dropout = tcn_dropout, 
                                   seq_last = False)
                            
        self.critic_tcn_1 = TCNBlock(input_size, tcn_layers, 
                                     kernel_size = tcn_kernel_size, 
                                     dropout = tcn_dropout, 
                                     seq_last = False)                            
        self.critic_tcn_2 = TCNBlock(input_size, tcn_layers, 
                                     kernel_size = tcn_kernel_size, 
                                     dropout = tcn_dropout, 
                                     seq_last = False)
                
        policy_layers = []
        critic_layers_1 = []
        critic_layers_2 = []
        
        input = tcn_layers[-1]
        for i, units in enumerate(hidden_layers):
            output = units
            policy_layers.append(nn.Linear(input, output))
            if i == 0:
                input += action_size
            critic_layers_1.append(nn.Linear(input, output))
            critic_layers_2.append(nn.Linear(input, output))
            input = output
            
            policy_layers.append(nn.ReLU())
            critic_layers_1.append(nn.ReLU())
            critic_layers_2.append(nn.ReLU())
            
            policy_layers.append(nn.Dropout(p = hidden_dropout))
            critic_layers_1.append(nn.Dropout(p = hidden_dropout))
            critic_layers_2.append(nn.Dropout(p = hidden_dropout))
            
        self.policy_layers = nn.Sequential(*policy_layers)
        self.critic_layers_1 = nn.Sequential(*critic_layers_1)
        self.critic_layers_2 = nn.Sequential(*critic_layers_2)
        
        self.policy_output = nn.Linear(input, action_size)
        self.critic_output_1 = nn.Linear(input, 1)
        self.critic_output_2 = nn.Linear(input, 1)
        
        self.critic_params_1 = (list(self.critic_layers_1.parameters()) 
                                + list(self.critic_output_1.parameters())
                                + list(self.critic_tcn_1.parameters()))
        self.critic_params_2 = (list(self.critic_layers_2.parameters()) 
                                + list(self.critic_output_2.parameters())
                                + list(self.critic_tcn_2.parameters()))
        
        self.policy_params = (list(self.policy_layers.parameters()) 
                              + list(self.policy_output.parameters())
                              + list(self.policy_tcn.parameters()))
                              
    def forward(self, obs, mask = None):
        x = obs
        if self.scale:
            x = self.feature_scaler(x)
            
        x = self.policy_tcn(x, mask = mask)
        x = x[:, :, -1] # get the last output in the sequence
        
        x = self.policy_layers(x)
        
        x = torch.tanh(self.policy_output(x))
        x = x * self.action_scale
        
        return x
        
    def action(self, obs, mask = None):
        return self.forward(obs, mask = mask)
        
    def critic_value(self, obs, action, mask = None):
        x1 = self.critic_value_1(obs, action, mask = mask)
        x2 = self.critic_value_2(obs, action, mask = mask)
        
        return x1, x2
        
    def critic_value_1(self, obs, action, mask = None):
        x = obs
        if self.scale:
            x = self.feature_scaler(x)
        
        x = self.critic_tcn_1(x, mask = mask)
        x = x[:, :, -1] # get the last output in the sequence
        
        x = torch.cat([x, action], dim = -1)
        x = self.critic_layers_1(x)
        
        x = self.critic_output_1(x)
        
        return x
        
    def critic_value_2(self, obs, action, mask = None):
        x = obs
        if self.scale:
            x = self.feature_scaler(x)
        
        x = self.critic_tcn_2(x, mask = mask)
        x = x[:, :, -1] # get the last output in the sequence
        
        x = torch.cat([x, action], dim = -1)
        x = self.critic_layers_2(x)
        
        x = self.critic_output_2(x)
        
        return x