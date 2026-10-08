import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Attention Mechanisms
# ==========================================

class ChannelAttention(nn.Module):
    """Channel Attention Module to recalibrate channel-wise feature responses."""
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
    """Spatial Attention Module to focus on informative regions of the feature map."""
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
    """Temporal Attention Module to weigh the importance of different time steps in the GRU output."""
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

# ==========================================
# 2. Main Backbone Network (Metric_ContinualNet)
# ==========================================

class Metric_ContinualNet(nn.Module):
    """
    Main Neural Network Backbone for EMG Prosthetic Control.
    Extracts robust embeddings for metric learning and predicts grip force.
    """
    def __init__(self, embedding_dim=32, proj_dim=128, num_sensors=12):
        super(Metric_ContinualNet, self).__init__()
        
        # Early Sensor Attention Gate
        self.sensor_attention = nn.Sequential(
            nn.Linear(num_sensors, num_sensors),
            nn.Sigmoid()
        )
        
        # Dilated Multi-Scale Temporal Convolutions
        self.branch1 = nn.Conv2d(1, 4, kernel_size=(1, 15), stride=(1, 4), padding=(0, 7))
        self.branch2 = nn.Conv2d(1, 4, kernel_size=(1, 9), stride=(1, 4), dilation=(1, 4), padding=(0, 16))
        self.branch3 = nn.Conv2d(1, 8, kernel_size=(1, 9), stride=(1, 4), dilation=(1, 8), padding=(0, 32))
        
        self.bn_temp = nn.BatchNorm2d(16)
        
        # Depthwise Separable Spatial Convolutions
        self.depthwise = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=0, groups=16)
        self.pointwise = nn.Conv2d(32, 32, kernel_size=1)
        self.bn_spat = nn.BatchNorm2d(32)
        
        # Attention Modules
        self.ca = ChannelAttention(32)
        self.sa = SpatialAttention()
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 4))
        self.dropout = nn.Dropout(0.4) 
        
        # GRU and Temporal Attention
        s_new = num_sensors // 2  
        gru_input_dim = 32 * s_new  
        
        self.gru = nn.GRU(
            input_size=gru_input_dim, 
            hidden_size=64,  
            num_layers=1, 
            batch_first=True
        )
        self.temporal_attention = TemporalAttention(64)
        
        # Output Heads
        self.embedding_head = nn.Sequential(
            nn.Linear(64, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, embedding_dim) 
        )

        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, proj_dim) 
        )

        self.regression_head = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (Batch, Time, Sensors)
            
        Returns:
            proj_embeddings: Normalized projected embeddings (for SupCon loss)
            base_embeddings: Normalized base embeddings (for SLDA/Prototypes)
            force: Predicted grip force (0.0 to 1.0)
        """
        B, W, S = x.shape
        
        # 1. Sensor Attention
        sensor_importance = self.sensor_attention(x.mean(dim=1)) # (B, S)
        x = x * sensor_importance.unsqueeze(1) 
        
        # 2. Reshape for 2D Convolutions: (B, 1, Sensors, Time)
        x = x.permute(0, 2, 1).contiguous()
        x = x.unsqueeze(1)
        
        # 3. Multi-Scale Branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        x = torch.cat([b1, b2, b3], dim=1) 
        x = F.relu(self.bn_temp(x))
        
        # 4. Spatial Convolutions
        x = F.pad(x, (1, 1, 0, 0), mode='constant', value=0) 
        x = F.pad(x, (0, 0, 1, 1), mode='circular')          
        
        x = self.depthwise(x)
        x = F.relu(self.bn_spat(self.pointwise(x)))
        
        # 5. Channel & Spatial Attention
        x = self.ca(x)
        x = self.sa(x)
        
        # 6. Pooling & Dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        # 7. Reshape for GRU: (B, Time_new, Features)
        B, C, S_new, T_new = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T_new, C * S_new)
        
        # 8. Temporal Processing (GRU + Attention)
        gru_out, _ = self.gru(x)
        features = self.temporal_attention(gru_out)
        
        # 9. Output Heads
        base_embeddings = self.embedding_head(features)
        proj_embeddings = self.projection_head(base_embeddings)
        
        # L2 Normalization for Metric Space
        base_embeddings = F.normalize(base_embeddings, p=2, dim=1)
        proj_embeddings = F.normalize(proj_embeddings, p=2, dim=1)
        
        # Force Regression
        force = self.regression_head(features)
        
        return proj_embeddings, base_embeddings, force

# ==========================================
# 3. Streaming Linear Discriminant Analysis (SLDA)
# ==========================================

class StreamingLDA:
    """
    Memory-efficient classifier that updates means and covariance incrementally 
    without backpropagation. Ideal for Edge AI and Continual Learning.
    """
    def __init__(self, input_dim=32, shrinkage_param=1e-4, device='cuda'):
        self.input_dim = input_dim
        self.shrinkage_param = shrinkage_param
        self.device = device
        
        # Dynamic dictionaries to handle incremental class additions
        self.class_means = {}
        self.class_counts = {}
        
        # Scatter matrix (unnormalized covariance matrix)
        self.scatter_matrix = torch.zeros((input_dim, input_dim)).to(device)
        self.total_samples = 0
        
        # Inverted covariance (Precision matrix) cached for fast inference
        self.precision_matrix = None

    def _update_scatter(self, features, labels):
        """Updates the scatter matrix and class means with new data."""
        unique_classes = torch.unique(labels)
        
        for c in unique_classes:
            c = c.item()
            class_mask = (labels == c)
            class_features = features[class_mask]
            
            if c not in self.class_means:
                self.class_means[c] = torch.zeros(self.input_dim).to(self.device)
                self.class_counts[c] = 0
            
            old_count = self.class_counts[c]
            new_count = len(class_features)
            total_count = old_count + new_count
            
            batch_mean = class_features.mean(dim=0)
            
            # 1. Update scatter matrix
            centered_features = class_features - batch_mean
            self.scatter_matrix += torch.matmul(centered_features.T, centered_features)
            
            # 2. Streaming update for the class mean
            self.class_means[c] = (self.class_means[c] * old_count + batch_mean * new_count) / total_count
            self.class_counts[c] = total_count
            self.total_samples += new_count
            
        self._compute_precision_matrix()

    def _compute_precision_matrix(self):
        """Computes and caches the inverted covariance (precision) matrix."""
        k = len(self.class_means)
        if k == 0 or self.total_samples <= k:
            return
            
        # Normalize scatter to get actual covariance
        covariance = self.scatter_matrix / (self.total_samples - k)
            
        # Apply Shrinkage for numerical stability
        identity = torch.eye(self.input_dim).to(self.device)
        cov_shrunk = (1 - self.shrinkage_param) * covariance + self.shrinkage_param * identity
        
        # Invert to get precision matrix
        self.precision_matrix = torch.linalg.inv(cov_shrunk)

    def fit_base(self, dataloader, backbone):
        """Fits the initial covariance matrix on base classes."""
        backbone.eval()
        print(" [Deep SLDA] Fitting initial Covariance Matrix on Base Classes...")
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, features, _ = backbone(inputs) 
                self._update_scatter(features, labels)
        print(f" [Deep SLDA] Fit complete. Tracking {len(self.class_means)} base classes.")

    def stream_new_data(self, dataloader, backbone):
        """Streams few-shot updates for novel classes without backpropagation."""
        backbone.eval()
        print(" [Deep SLDA] Streaming Few-Shot update for Novel Classes...")
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, features, _ = backbone(inputs)
                self._update_scatter(features, labels)
        print(f" [Deep SLDA] Update complete. Total tracked classes: {len(self.class_means)}.")

    def predict(self, features):
        """
        Predicts class labels using the cached precision matrix.
        
        Args:
            features: Tensor of shape (Batch, input_dim)
            
        Returns:
            predicted_classes: Tensor of predicted class labels
            max_scores: Tensor of maximum decision scores (useful for uncertainty)
        """
        if self.precision_matrix is None:
            raise ValueError("Model not fitted yet. Call fit_base() first.")
            
        classes = list(self.class_means.keys())
        M = torch.stack([self.class_means[c] for c in classes])
        
        # Calculate weights: W = Precision @ M.T
        W = torch.matmul(self.precision_matrix, M.T)
        
        # Calculate biases: b = -0.5 * diag(M @ Precision @ M.T)
        b = -0.5 * torch.sum(M * torch.matmul(M, self.precision_matrix), dim=1)
        
        # Linear scoring: Scores = X @ W + b
        scores = torch.matmul(features, W) + b
        
        max_scores, preds_idx = torch.max(scores, dim=1)
        predicted_classes = torch.tensor([classes[i] for i in preds_idx]).to(self.device)
        
        return predicted_classes, max_scores