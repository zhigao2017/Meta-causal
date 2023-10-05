
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class mapping(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim = 512, out_dim=1024, layernum=4):
        ''' 
        '''
        super().__init__()
        self.layernum = layernum
        if layernum == 4:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, out_dim)
        elif layernum == 2:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, out_dim)            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): 
        ''' x '''
        if self.layernum == 4:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
        elif self.layernum == 2:
            x = self.relu(self.fc1(x))
            x = self.fc2(x)            
        return x


class effect_to_weight(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim = 256, out_dim = 1, layernum=2, hidden_dim2 = 128):
        ''' 
        '''
        super().__init__()
        
        self.layernum = layernum
        if layernum == 2:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, out_dim)
        elif layernum == 3:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim2)            
            self.fc3 = nn.Linear(hidden_dim2, out_dim)  
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): 
        ''' x '''
        if self.layernum == 2:
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
        return x


