# train_offline.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from architecture import Metric_ContinualNet
from dataset_handler import base_train_loader, base_test_loader, target_train_loader
from continual_algorithms import coral_loss


def mine_triplets_in_batch(embeddings, labels):
    anchors, positives, negatives = [], [], []
    for i in range(len(labels)):
        anchor, label = embeddings[i], labels[i]
        
        pos_mask = (labels == label)
        pos_mask[i] = False
        if pos_mask.sum() == 0: pos_mask[i] = True
        pos_idx = torch.where(pos_mask)[0]
        positive = embeddings[pos_idx[torch.randint(0, len(pos_idx), (1,))]]
        
        neg_mask = (labels != label)
        if neg_mask.sum() == 0: continue
        neg_idx = torch.where(neg_mask)[0]
        negative = embeddings[neg_idx[torch.randint(0, len(neg_idx), (1,))]]
        
        anchors.append(anchor)
        positives.append(positive.squeeze(0))
        negatives.append(negative.squeeze(0))
        
    if len(anchors) == 0: return embeddings, embeddings, embeddings
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

def train_metric_learning(model, train_loader, epochs=15, lr=0.001, device='cuda'):
    criterion_metric = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion_reg = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            actual_force = inputs.abs().mean(dim=(1,2)).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            embeddings, predicted_force = model(inputs)
            anchors, positives, negatives = mine_triplets_in_batch(embeddings, labels)
            
            loss_metric = criterion_metric(anchors, positives, negatives)
            loss_reg = criterion_reg(predicted_force, actual_force)
            
            loss = loss_metric + (0.5 * loss_reg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        print(f"Base Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader.dataset):.4f}")
    return model


def train_domain_adaptation_fixed(model, original_model, source_loader, target_loader, epochs=10, lambda_coral=0.5, device='cuda'):
   
    model.to(device)
    original_model.to(device)
    original_model.eval() 
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print("\n Starting FIXED Domain Adaptation (Anchored CORAL)...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        coral_epoch_loss = 0.0
        anchor_epoch_loss = 0.0
        
        for (src_x, _), (tgt_x, _) in zip(source_loader, target_loader):
            src_x, tgt_x = src_x.to(device), tgt_x.to(device)
            optimizer.zero_grad()
            
            src_features, _ = model(src_x)
            tgt_features, _ = model(tgt_x)
            
            with torch.no_grad():
                orig_src_features, _ = original_model(src_x)
                
            
            loss_anchor = F.mse_loss(src_features, orig_src_features)
            loss_coral = coral_loss(src_features, tgt_features)
            
            total_loss = loss_anchor + (lambda_coral * loss_coral)
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            coral_epoch_loss += loss_coral.item()
            anchor_epoch_loss += loss_anchor.item()
            
        print(f"DOA Epoch [{epoch+1}/{epochs}] | Anchor Loss: {anchor_epoch_loss:.4f} | CORAL Loss: {coral_epoch_loss:.4f}")
    return model



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if base_train_loader and target_train_loader:
        print(" Phase 1: Training Baseline Backbone...")
        metric_model = Metric_ContinualNet(embedding_dim=32).to(device)
        metric_model = train_metric_learning(metric_model, base_train_loader, epochs=15, device=device)
        
        
        torch.save(metric_model.state_dict(), "baseline_backbone.pth")
        
        print("\n Phase 2: Applying Domain Adaptation (DOA)...")
        doa_backbone = copy.deepcopy(metric_model)
        doa_backbone = train_domain_adaptation_fixed(
            doa_backbone, 
            metric_model, 
            base_train_loader, 
            target_train_loader, 
            epochs=10, 
            lambda_coral=0.5, 
            device=device
        )
        
        
        torch.save(doa_backbone.state_dict(), "trained_doa_backbone.pth")
        print(" Training completed! Final robust weights saved as 'trained_doa_backbone.pth'.")
    else:
        print(" Error: Data not found. Check dataset paths in dataset_handler.py")