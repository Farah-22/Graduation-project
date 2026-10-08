import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch.utils.data import ConcatDataset, DataLoader

from src.utils import plot_confusion_matrix

# ==========================================
# 1. Core Evaluation Helpers
# ==========================================

def compute_class_prototypes(model, dataloader, device='cuda'):
    """Computes class prototypes (mean embeddings) for metric-based inference."""
    model.eval()
    class_embeddings = {}
    with torch.no_grad():
        for inputs, labels in dataloader:
            _, base_emb, _ = model(inputs.to(device)) 
            for i, label in enumerate(labels.cpu().numpy()):
                if label not in class_embeddings: class_embeddings[label] = []
                class_embeddings[label].append(base_emb[i].cpu())
                
    prototypes = {}
    for label, embs in class_embeddings.items():
        stacked_embs = torch.stack(embs)
        proto_mean = stacked_embs.mean(dim=0)
        prototypes[label] = F.normalize(proto_mean.unsqueeze(0), p=2, dim=1).squeeze(0).to(device)
    return prototypes

def collect(model, loader, prototypes, device='cuda'):
    """Collects predictions, true labels, and distances using Prototypes."""
    model.eval()
    all_preds, all_true, all_dist = [], [], []
    sorted_keys = sorted(prototypes.keys())
    proto_tensor = torch.stack([prototypes[k] for k in sorted_keys]).to(device)

    with torch.no_grad():
        for x, y in loader:
            _, embeddings, _ = model(x.to(device))
            distances = torch.cdist(embeddings, proto_tensor, p=2.0)
            min_dist, pred_indices = torch.min(distances, dim=1)
            
            preds = [sorted_keys[idx] for idx in pred_indices.cpu().numpy()]
            all_preds.extend(preds)
            all_true.extend(y.numpy())
            all_dist.extend(min_dist.cpu().numpy())
            
    return np.array(all_preds), np.array(all_true), np.array(all_dist)

def causal_majority_vote(pred, true, k):
    """
    Applies Segment-Aware Causal Majority Vote smoothing.
    Prevents looking into the future (causal) and resets at segment boundaries.
    """
    out = pred.copy()
    seg_start = 0
    for i in range(len(pred)):
        if i > 0 and true[i] != true[i-1]: 
            seg_start = i 
        out[i] = np.bincount(pred[max(seg_start, i-k+1): i+1]).argmax()
    return out

# ==========================================
# 2. Ablation Study Evaluation
# ==========================================

def evaluate_condition(name, use_adabn, model, cal_loader, test_loader, device='cuda'):
    """Evaluates a specific condition (Raw vs AdaBN) for Ablation Study."""
    from src.calibration import adapt_bn
    import copy
    
    eval_model = copy.deepcopy(model)
    if use_adabn: 
        eval_model = adapt_bn(eval_model, cal_loader, device)
        
    protos = compute_class_prototypes(eval_model, cal_loader, device=device)
    p, t, _ = collect(eval_model, test_loader, protos, device)
    
    mv = t != 0
    bal_acc = 100 * balanced_accuracy_score(t, p)
    move_acc = 100 * np.mean(p[mv] == t[mv])
    false_act = 100 * np.mean(p[t == 0] != 0)
    
    print(f"{name:20s} | Balanced: {bal_acc:.2f}% | Movement: {move_acc:.2f}% | False-Act: {false_act:.2f}%")
    return p, t 

# ==========================================
# 3. Continual Learning Evaluation (Prototypes)
# ==========================================

def evaluate_all_tasks(model, loader_t1, loader_t2, prototypes, k_smoothing=20, threshold=0.8, device='cuda'):
    """Evaluates the model on Task 1 + Task 2 using Prototypes and Safe Rejection."""
    model.eval()
    combined_dataset = ConcatDataset([loader_t1.dataset, loader_t2.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)
    
    sorted_keys = sorted(prototypes.keys())
    proto_tensor = torch.stack([prototypes[k] for k in sorted_keys]).to(device)
    
    all_preds_raw, all_true, all_dist = [], [], []
    
    with torch.no_grad():
        for x, y in combined_loader:
            x, y = x.to(device), y.to(device)
            _, embeddings, _ = model(x)
            distances = torch.cdist(embeddings, proto_tensor, p=2.0)
            min_dist, predicted_indices = torch.min(distances, dim=1)
            
            preds = [sorted_keys[idx] for idx in predicted_indices.cpu().numpy()]
            all_preds_raw.extend(preds)
            all_true.extend(y.cpu().numpy())
            all_dist.extend(min_dist.cpu().numpy())
            
    all_preds_raw = np.array(all_preds_raw)
    all_true = np.array(all_true)
    all_dist = np.array(all_dist)
    
    pred_smoothed = causal_majority_vote(all_preds_raw, all_true, k=k_smoothing)
    
    # Safe Rejection System
    final_preds = []
    for i in range(len(pred_smoothed)):
        if all_dist[i] > threshold:
            final_preds.append(99) # Rejected
        else:
            final_preds.append(pred_smoothed[i])
            
    final_preds = np.array(final_preds)

    # Plotting and Reporting
    actual_classes = [0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 99]
    class_names = ['Rest(0)'] + [f'Mov{i}' for i in range(18, 29)] + ['Rejected']
    
    plot_confusion_matrix(all_true, final_preds, actual_classes, class_names, 
                          f"Universal Metric Space: Task 1 + Task 2 (Smoothed k={k_smoothing})")

    print("\n" + "="*65)
    print(" CONTINUAL LEARNING CLASSIFICATION REPORT ")
    print("="*65)
    print(classification_report(all_true, final_preds, labels=actual_classes, 
                                target_names=class_names, zero_division=0))

# ==========================================
# 4. Streaming LDA Evaluation
# ==========================================

def evaluate_slda_all_tasks(slda, model, loader_t1, loader_t2, k_smoothing=20, device='cuda'):
    """Evaluates the model on Task 1 + Task 2 using Streaming LDA."""
    model.eval()
    combined_dataset = ConcatDataset([loader_t1.dataset, loader_t2.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)
    
    all_preds_raw, all_true = [], []
    
    with torch.no_grad():
        for x, y in combined_loader:
            x, y = x.to(device), y.to(device)
            _, features, _ = model(x)
            preds, _ = slda.predict(features)
            all_preds_raw.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())
            
    all_preds_raw = np.array(all_preds_raw)
    all_true = np.array(all_true)
    
    pred_smoothed = causal_majority_vote(all_preds_raw, all_true, k=k_smoothing)
    
    mv_mask = all_true != 0
    acc_raw = 100 * np.mean(all_preds_raw[mv_mask] == all_true[mv_mask])
    acc_smooth = 100 * np.mean(pred_smoothed[mv_mask] == all_true[mv_mask])
    false_act = 100 * np.mean(pred_smoothed[all_true == 0] != 0)
    
    print("\n" + "="*60)
    print(" STREAMING LDA CLINICAL METRICS (Task 1 + Task 2) ")
    print("="*60)
    print(f"Movement Accuracy (Raw SLDA):         {acc_raw:.2f}%")
    print(f"Movement Accuracy (Smoothed k={k_smoothing}): {acc_smooth:.2f}%")
    print(f"False Activations during Rest:        {false_act:.2f}%")
    print("="*60)

    actual_classes = [0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    class_names = ['Rest(0)'] + [f'Mov{i}' for i in range(18, 29)]
    
    plot_confusion_matrix(all_true, pred_smoothed, actual_classes, class_names, 
                          f"Streaming LDA Classifier: Task 1 + Task 2 (Smoothed k={k_smoothing})", cmap='Greens')

# ==========================================
# 5. MC Dropout Uncertainty Quantification
# ==========================================

def enable_dropout(model):
    """Enables dropout layers during inference for MC Dropout."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_inference(model, slda, loader_t1, loader_t2, num_passes=5, uncertainty_threshold=0.1, device='cuda'):
    """Evaluates uncertainty using MC Dropout on Task 1 + Task 2."""
    model.eval()
    enable_dropout(model) 
    
    combined_dataset = ConcatDataset([loader_t1.dataset, loader_t2.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)
    
    total_samples = 0
    rejected_samples = 0
    accepted_correct = 0
    
    with torch.no_grad():
        for inputs, labels in combined_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            pass_predictions = torch.zeros(num_passes, batch_size, device=device)
            
            for p in range(num_passes):
                _, features, _ = model(inputs)
                preds, _ = slda.predict(features)
                pass_predictions[p] = preds
                
            for i in range(batch_size):
                total_samples += 1
                true_label = labels[i].item()
                sample_preds = pass_predictions[:, i].float()
                uncertainty = torch.var(sample_preds).item()
                final_pred = torch.mode(sample_preds).values.item()
                
                if uncertainty > uncertainty_threshold:
                    rejected_samples += 1
                else:
                    if final_pred == true_label:
                        accepted_correct += 1
                        
    accepted_samples = total_samples - rejected_samples
    accuracy = (accepted_correct / accepted_samples) * 100 if accepted_samples > 0 else 0
    
    print("\n" + "="*55)
    print(" UNCERTAINTY QUANTIFICATION (MC DROPOUT) REPORT ")
    print("="*55)
    print(f"Total Tested Samples:               {total_samples}")
    print(f"Rejected (Uncertain / High Risk):    {rejected_samples} ({(rejected_samples/total_samples)*100:.2f}%)")
    print(f"Accepted (Confident Actions):       {accepted_samples} ({(accepted_samples/total_samples)*100:.2f}%)")
    print(f"Accuracy on Confident Actions:      {accuracy:.2f}%")
    print("="*55)

# ==========================================
# 6. Domain Adaptation (DOA) Evaluation
# ==========================================

def test_doa_vs_baseline(baseline_model, doa_model, slda, dataloader, device='cuda'):
    """Compares Baseline vs. DOA model on Target Domain (Shifted Data)."""
    baseline_model.eval()
    doa_model.eval()

    baseline_correct = 0
    doa_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            _, emb_base, _ = baseline_model(x)
            preds_base, _ = slda.predict(emb_base)
            baseline_correct += (preds_base == y).sum().item()

            _, emb_doa, _ = doa_model(x)
            preds_doa, _ = slda.predict(emb_doa)
            doa_correct += (preds_doa == y).sum().item()

            total += y.size(0)

    base_acc = (baseline_correct / total) * 100
    doa_acc = (doa_correct / total) * 100

    print("\n" + "="*60)
    print(" Comparing Baseline vs. DOA on Target Domain (Shifted Data)")
    print("="*60)
    print(f"Total Test Samples (Target Domain):          {total}")
    print(f"Baseline Model Accuracy (Without Adaptation): {base_acc:.2f}%")
    print(f"DOA Model Accuracy (With Anchored CORAL):     {doa_acc:.2f}%")
    print("="*60)
    
    if doa_acc > base_acc:
        print("✅ Domain Adaptation Successful! DOA model is more robust to noise/shift.")
    else:
        print("⚠️ DOA model did not outperform the baseline.")

def test_doa_on_clean_data(doa_model, baseline_model, slda, loader_t1, loader_t2, device='cuda'):
    """Evaluates DOA vs Baseline on CLEAN Test Data to check for catastrophic forgetting."""
    doa_model.eval()
    baseline_model.eval()

    combined_dataset = ConcatDataset([loader_t1.dataset, loader_t2.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)

    doa_correct = 0
    baseline_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in combined_loader:
            x, y = x.to(device), y.to(device)

            _, emb_base, _ = baseline_model(x)
            preds_base, _ = slda.predict(emb_base)
            baseline_correct += (preds_base == y).sum().item()

            _, emb_doa, _ = doa_model(x)
            preds_doa, _ = slda.predict(emb_doa)
            doa_correct += (preds_doa == y).sum().item()

            total += y.size(0)

    base_acc = (baseline_correct / total) * 100
    doa_acc = (doa_correct / total) * 100

    print("\n" + "="*60)
    print(" DOA vs Baseline on CLEAN Data (Task 1 + Task 2)")
    print("="*60)
    print(f"Total Clean Test Samples:                      {total}")
    print(f"Baseline Model Accuracy on Clean Data:         {base_acc:.2f}%")
    print(f"DOA Model Accuracy on Clean Data:              {doa_acc:.2f}%")
    print("="*60)
    
    diff = abs(base_acc - doa_acc)
    if diff < 3.0:
        print("✅ Excellent! Anchor Loss successfully preserved performance on clean data.")
    else:
        print("⚠️ Noticeable trade-off detected between noise-robustness and clean accuracy.")