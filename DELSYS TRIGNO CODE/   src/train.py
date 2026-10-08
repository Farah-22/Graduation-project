import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.losses import SupConLoss, coral_loss
from src.calibration import freeze_bn

# ==========================================
# 1. Helper Functions for Training
# ==========================================

def prototype_accuracy(model, support_loader, query_loader, device, max_support_batches=60):
    """
    Calculates the accuracy of the model using Prototypes (used as a validation metric 
    during Phase 1 training).
    """
    model.eval()
    embs, lbls = [], []
    
    # 1. Build Prototypes from Support (Training) Data
    for b, (x, y) in enumerate(support_loader):
        if b >= max_support_batches: break
        _, base, _ = model(x.to(device))
        embs.append(base.cpu())
        lbls.append(y)
        
    embs = torch.cat(embs)
    lbls = torch.cat(lbls)
    keys = torch.unique(lbls).tolist()
    
    protos = torch.stack([embs[lbls == k].mean(0) for k in keys])
    protos = F.normalize(protos, p=2, dim=1).to(device)
    keys_t = torch.tensor(keys, device=device)

    # 2. Evaluate Query (Validation) Data based on training centers
    correct = total = 0
    for x, y in query_loader:
        _, base, _ = model(x.to(device))
        dists = torch.cdist(base, protos)
        pred = keys_t[dists.argmin(1)]
        correct += (pred == y.to(device)).sum().item()
        total += y.size(0)
        
    return 100.0 * correct / max(1, total)


# ==========================================
# 2. Phase 1: Supervised Contrastive Training
# ==========================================

def train_supcon_phase1(model, train_loader, val_loader, epochs=70, lr=0.001, device='cuda'):
    """
    Phase 1 Training: Constructs the metric space using Supervised Contrastive Learning.
    Uses Channel-wise Warping for data augmentation and monitors true validation accuracy.
    """
    criterion = SupConLoss(temperature=0.07) 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # Scheduler monitors Accuracy and reduces LR if it plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    print("\n [Phase 1] Training with Strict Validation & Channel-wise Warping...")
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Channel-wise Warping: Independent scaling for each sensor
            warp_factor = torch.empty(inputs.shape[0], 1, inputs.shape[2]).uniform_(0.8, 1.2).to(device)
            augmented_inputs = inputs * warp_factor
            noise = torch.randn_like(augmented_inputs) * 0.01
            augmented_inputs = augmented_inputs + noise
            
            optimizer.zero_grad()
            proj_emb, base_emb, _ = model(augmented_inputs)
            loss = criterion(proj_emb, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
            
        epoch_train_loss = running_train_loss / max(1, train_samples)
        
        # Calculate True Validation Accuracy using Prototypes
        val_accuracy = prototype_accuracy(model, train_loader, val_loader, device)
            
        scheduler.step(val_accuracy) 
        
        marker = ""
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            marker = " (New High!)"
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{epochs}] | LR: {current_lr:.6f} | Loss: {epoch_train_loss:.4f} | Val Acc: {val_accuracy:.2f}% {marker}")
        
    print(f"\n Phase 1 Completed! Best True Validation Accuracy: {best_val_acc:.2f}%")
    model.load_state_dict(best_model_wts)
    return model


# ==========================================
# 3. Phase 2: Few-Shot Fine-Tuning
# ==========================================

def few_shot_finetune(model, train_loader, val_loader, epochs=10, lr=0.0001, device='cuda'):
    """
    Phase 2 Training: Few-Shot Fine-Tuning for a new subject or task.
    Freezes early convolutional branches to prevent catastrophic forgetting.
    """
    print("\n [Phase 2] Few-Shot Fine-Tuning in progress...")
    
    # Freeze early multi-scale branches
    for param in model.branch1.parameters(): param.requires_grad = False
    for param in model.branch2.parameters(): param.requires_grad = False
    for param in model.branch3.parameters(): param.requires_grad = False
    
    # Freeze BatchNorm layers to preserve robust domain statistics
    model = freeze_bn(model)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = SupConLoss(temperature=0.07)
    
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Data Augmentation
            warp_factor = torch.empty(x.shape[0], 1, x.shape[2]).uniform_(0.8, 1.2).to(device)
            augmented_x = x * warp_factor + (torch.randn_like(x) * 0.01).to(device)
            
            optimizer.zero_grad()
            proj_emb, _, _ = model(augmented_x)
            loss = criterion(proj_emb, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            
        val_acc = prototype_accuracy(model, train_loader, val_loader, device)
        print(f"Fine-tune Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader.dataset):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_wts)
    return model


# ==========================================
# 4. Unsupervised Domain Adaptation (UDA)
# ==========================================

def train_domain_adaptation_fixed(model, original_model, source_loader, target_loader, epochs=10, lambda_coral=0.5, device='cuda'):
    """
    Domain Adaptation Training Loop: Uses Knowledge Distillation (Anchor Loss) 
    to prevent Feature Collapse while aligning domains with CORAL loss.
    """
    model.to(device)
    original_model.to(device)
    original_model.eval() # Freeze the original robust model
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    
    print("\n Starting Domain Adaptation (Anchored CORAL)...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        coral_epoch_loss = 0.0
        anchor_epoch_loss = 0.0
        
        for (src_x, _), (tgt_x, _) in zip(source_loader, target_loader):
            src_x, tgt_x = src_x.to(device), tgt_x.to(device)
            
            optimizer.zero_grad()
            
            # Extract features
            _, src_features, _ = model(src_x)
            _, tgt_features, _ = model(tgt_x)
            
            with torch.no_grad():
                _, orig_src_features, _ = original_model(src_x)
                
            # Anchor Loss: Preserves old knowledge
            loss_anchor = F.mse_loss(src_features, orig_src_features)
            
            # CORAL Loss: Aligns distributions with shifted/noisy data
            loss_coral = coral_loss(src_features, tgt_features)
            
            total_loss = loss_anchor + (lambda_coral * loss_coral)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            coral_epoch_loss += loss_coral.item()
            anchor_epoch_loss += loss_anchor.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Anchor: {anchor_epoch_loss:.4f} | CORAL: {coral_epoch_loss:.4f}")
        
    return model