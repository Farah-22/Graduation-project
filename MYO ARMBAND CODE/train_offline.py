# train_offline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size, device=device).view(-1, 1), 0)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos
        
        valid_mask = (mask.sum(1) > 0)
        return loss[valid_mask].mean() if valid_mask.sum() > 0 else loss.mean()

def calculate_val_accuracy(model, val_loader, device='cuda'):
    model.eval()
    all_embs, all_labels = [], []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            embs, _ = model(x)
            all_embs.append(embs.cpu())
            all_labels.append(y.cpu())
            
    all_embs = torch.cat(all_embs).to(device)
    all_labels = torch.cat(all_labels).to(device)
    
    unique_labels = torch.unique(all_labels)
    protos = torch.stack([all_embs[all_labels == lbl].mean(0) for lbl in unique_labels])
    protos = F.normalize(protos, p=2, dim=1)
    
    dists = torch.cdist(all_embs, protos)
    preds = unique_labels[dists.argmin(1)]
    
    acc = (preds == all_labels).float().mean().item() * 100
    return acc

def train_metric_learning(model, train_loader, val_loader, epochs=30, lr=0.001, device='cuda', patience=7):
    criterion_metric = SupConLoss(temperature=0.07)
    criterion_reg = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print("\n" + "="*70)
    print("Starting Advanced Metric Learning (SupCon + Augmentation + Early Stopping)")
    print("="*70)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            warp_factor = torch.empty(inputs.shape[0], 1, inputs.shape[2]).uniform_(0.8, 1.2).to(device)
            augmented_inputs = inputs * warp_factor
            noise = torch.randn_like(augmented_inputs) * 0.01
            augmented_inputs = augmented_inputs + noise
            
            actual_force = augmented_inputs.abs().mean(dim=(1,2)).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            embeddings, predicted_force = model(augmented_inputs)
            
            loss_metric = criterion_metric(embeddings, labels)
            loss_reg = criterion_reg(predicted_force, actual_force)
            loss = loss_metric + (0.5 * loss_reg)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
            
        avg_train_loss = running_train_loss / max(1, train_samples)
        
        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for v_inputs, v_labels in val_loader:
                v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                v_force = v_inputs.abs().mean(dim=(1,2)).unsqueeze(1).to(device)
                
                v_embs, v_pred_force = model(v_inputs)
                
                v_loss_metric = criterion_metric(v_embs, v_labels)
                v_loss_reg = criterion_reg(v_pred_force, v_force)
                v_loss = v_loss_metric + (0.5 * v_loss_reg)
                
                running_val_loss += v_loss.item() * v_inputs.size(0)
                val_samples += v_inputs.size(0)
                
        avg_val_loss = running_val_loss / max(1, val_samples)
        val_accuracy = calculate_val_accuracy(model, val_loader, device)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " (New Best!)"
        else:
            patience_counter += 1
            marker = ""
            if patience_counter >= patience:
                print(f"\nEarly Stopping triggered at Epoch {epoch+1} (No improvement for {patience} epochs)")
                break
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{epochs}] | LR: {current_lr:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}% {marker}")
        
    print("\nTraining Completed!")
    model.load_state_dict(best_model_wts)
    return model

def compute_class_prototypes(model, dataloader, num_classes=3, device='cuda'):
    model.eval()
    class_embeddings = {i: [] for i in range(num_classes)}
    
    print("Extracting embeddings to compute prototypes...")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            embeddings, _ = model(inputs)
            
            for i in range(len(labels)):
                label = labels[i].item()
                if label in class_embeddings:
                    class_embeddings[label].append(embeddings[i].cpu())
                    
    prototypes = {}
    for label, embs in class_embeddings.items():
        if len(embs) > 0:
            stacked_embs = torch.stack(embs)
            proto_mean = stacked_embs.mean(dim=0)
            proto_normalized = F.normalize(proto_mean.unsqueeze(0), p=2, dim=1).squeeze(0)
            
            prototypes[label] = proto_normalized.to(device)
            print(f"Prototype for Class {label} computed successfully. (Shape: {proto_normalized.shape})")
        else:
            print(f"Warning: No samples found for Class {label}.")
            
    return prototypes

def few_shot_finetune_with_anchor(model, novel_loader, base_loader, 
                                   epochs=20, lr=0.0001, 
                                   lambda_anchor=5.0, device='cuda'):
    from itertools import cycle
    
    anchor_model = copy.deepcopy(model)
    anchor_model.eval()
    for param in anchor_model.parameters():
        param.requires_grad = False
    
    model = copy.deepcopy(model)
    
    for param in model.temporal_filter.parameters():
        param.requires_grad = False
    for param in model.depthwise.parameters():
        param.requires_grad = False
    for param in model.pointwise.parameters():
        param.requires_grad = False
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)
    
    criterion_metric = SupConLoss(temperature=0.07)
    criterion_anchor = nn.MSELoss()
    
    print("\n" + "="*70)
    print("Few-Shot Fine-Tuning with PROPER Anchor Loss")
    print("="*70)
    print(f"Novel Samples: {len(novel_loader.dataset)}")
    print(f"Base Samples (for Anchor): {len(base_loader.dataset)}")
    print(f"Anchor Loss Weight: {lambda_anchor}")
    print(f"Anchor Model: FROZEN (won't update)")
    print("="*70)
    
    for epoch in range(epochs):
        model.train()
        running_novel_loss = 0.0
        running_anchor_loss = 0.0
        total_loss_epoch = 0.0
        samples = 0
        
        base_loader_cycle = cycle(base_loader)
        
        for x_novel, y_novel in novel_loader:
            x_novel, y_novel = x_novel.to(device), y_novel.to(device)
            
            x_base, y_base = next(base_loader_cycle)
            x_base, y_base = x_base.to(device), y_base.to(device)
            
            warp = torch.empty(x_novel.shape[0], 1, x_novel.shape[2]).uniform_(0.8, 1.2).to(device)
            x_novel_aug = x_novel * warp + torch.randn_like(x_novel) * 0.01
            
            optimizer.zero_grad()
            
            embs_novel, _ = model(x_novel_aug)
            loss_novel = criterion_metric(embs_novel, y_novel)
            
            embs_current_base, _ = model(x_base)
            
            with torch.no_grad():
                embs_anchor_base, _ = anchor_model(x_base)
            
            loss_anchor = criterion_anchor(embs_current_base, embs_anchor_base)
            
            total_loss = loss_novel + (lambda_anchor * loss_anchor)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_novel_loss += loss_novel.item() * x_novel.size(0)
            running_anchor_loss += loss_anchor.item() * x_base.size(0)
            total_loss_epoch += total_loss.item() * x_novel.size(0)
            samples += x_novel.size(0)
        
        avg_novel_loss = running_novel_loss / max(1, samples)
        avg_anchor_loss = running_anchor_loss / max(1, samples)
        avg_total_loss = total_loss_epoch / max(1, samples)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Total: {avg_total_loss:.4f} | "
              f"Novel: {avg_novel_loss:.4f} | Anchor: {avg_anchor_loss:.4f} | "
              f"Effective Anchor: {avg_anchor_loss * lambda_anchor:.4f}")
    
    print("\nFew-Shot Fine-Tuning with PROPER Anchor Completed!")
    return model