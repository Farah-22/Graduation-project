import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def adapt_bn(model: nn.Module, loader: DataLoader, device: torch.device) -> nn.Module:
    """
    Adaptive Batch Normalization (AdaBN).
    
    Updates the running mean and variance of BatchNorm layers using 
    a small amount of target domain data (Few-Shot calibration).
    This is crucial for aligning the model's internal statistics 
    with the new subject or sensor conditions before inference.
    
    Args:
        model: The neural network model (e.g., Metric_ContinualNet).
        loader: DataLoader containing the calibration/target data.
        device: The device to run the computation on ('cuda' or 'cpu').
        
    Returns:
        The model with updated BatchNorm statistics, set to eval mode.
    """
    model.eval()
    
    # Identify all BatchNorm layers in the model
    bns = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    
    # Temporarily switch BatchNorm layers to train mode to update running stats
    # and disable momentum to compute exact batch statistics
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # Use batch statistics instead of running average
        m.train()
        
    # Forward pass to update the statistics
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device))
            
    # Revert BatchNorm layers back to eval mode and restore default momentum
    for m in bns:
        m.momentum = 0.1 
        
    model.eval()
    return model


def freeze_bn(model: nn.Module) -> nn.Module:
    """
    Freezes Batch Normalization layers.
    
    Useful during Few-Shot Fine-Tuning to prevent the model from 
    overwriting the robust domain statistics learned during pre-training.
    
    Args:
        model: The neural network model.
        
    Returns:
        The model with frozen BatchNorm layers.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            # Disable gradient updates for BN parameters
            for param in m.parameters():
                param.requires_grad = False
    return model