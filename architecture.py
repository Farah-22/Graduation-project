# architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
     def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, gru_output):
        attention_weights = self.attention(gru_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * gru_output, dim=1)
        return context_vector

class Metric_ContinualNet(nn.Module):
    def __init__(self, embedding_dim=32, num_sensors=8, window_size=200):
        super(Metric_ContinualNet, self).__init__()
        
        
        self.temporal_filter = nn.Conv2d(1, 16, kernel_size=(1, 15), stride=(1, 2), padding=(0, 7))
        self.bn_temp = nn.BatchNorm2d(16)
        
        self.depthwise = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=0, groups=16)
        self.pointwise = nn.Conv2d(32, 32, kernel_size=1)
        self.bn_spat = nn.BatchNorm2d(32)
        
        self.ca = ChannelAttention(32)
        self.sa = SpatialAttention()
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout = nn.Dropout(0.3)
        
        gru_input_dim = 32 * 4 
        
        self.gru = nn.GRU(
            input_size=gru_input_dim, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True,
            dropout=0.3
        )
        self.temporal_attention = TemporalAttention(64)
        
        
        self.embedding_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, embedding_dim)
        )
        
        
        self.regression_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, W, S = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.unsqueeze(1)
        
        x = F.relu(self.bn_temp(self.temporal_filter(x)))
        x = F.pad(x, (1, 1, 0, 0), mode='constant', value=0) 
        x = F.pad(x, (0, 0, 1, 1), mode='circular')          
        
        x = self.depthwise(x)
        x = F.relu(self.bn_spat(self.pointwise(x)))
        x = self.ca(x)
        x = self.sa(x)
        
        x = self.pool(x)
        x = self.dropout(x)
        
        B, C, S_new, T_new = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T_new, C * S_new)
        
        gru_out, _ = self.gru(x)
        features = self.temporal_attention(gru_out)
        
        
        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1) 
        
        
        force = self.regression_head(features)
        
        return embeddings, force