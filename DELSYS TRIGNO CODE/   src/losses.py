import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Supervised Contrastive Loss (SupCon)
# ==========================================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Pulls together embeddings of the same class and pushes apart embeddings 
    of different classes in the metric space.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Forward pass for SupCon Loss.
        
        Args:
            features: Tensor of shape (batch_size, feature_dim).
            labels: Tensor of shape (batch_size,).
            
        Returns:
            Scalar loss value.
        """
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        # Mask to identify positive pairs (same class)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-comparison
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size, device=device).view(-1, 1), 
            0
        )
        mask = mask * logits_mask
        
        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos
        
        # Handle edge case where a sample has no positive pairs in the batch
        valid_mask = (mask.sum(1) > 0)
        
        return loss[valid_mask].mean() if valid_mask.sum() > 0 else loss.mean()


# ==========================================
# 2. Deep CORAL Loss (Domain Adaptation)
# ==========================================

def coral_loss(source_features, target_features):
    """
    Deep CORAL (Correlation Alignment) Loss.
    Aligns the second-order statistics (covariances) of the source and target domains.
    Crucial for Unsupervised Domain Adaptation (UDA) to handle sensor shifts.
    
    Args:
        source_features: Tensor of shape (batch_size, feature_dim).
        target_features: Tensor of shape (batch_size, feature_dim).
        
    Returns:
        Scalar CORAL loss value.
    """
    d = source_features.size(1)
    
    # Source covariance
    source_mean = torch.mean(source_features, 0, keepdim=True)
    source_centered = source_features - source_mean
    source_cov = (source_centered.t() @ source_centered) / (source_features.size(0) - 1)
    
    # Target covariance
    target_mean = torch.mean(target_features, 0, keepdim=True)
    target_centered = target_features - target_mean
    target_cov = (target_centered.t() @ target_centered) / (target_features.size(0) - 1)
    
    # Frobenius norm squared of the difference
    loss = torch.sum(torch.pow(source_cov - target_cov, 2))
    loss = loss / (4 * d * d)
    
    return loss