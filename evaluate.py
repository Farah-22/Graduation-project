# evaluate.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np


from architecture import Metric_ContinualNet
from continual_algorithms import StreamingLDA, compute_class_prototypes
from dataset_handler import t1_test_loader, t2_test_loader, target_test_loader

def distance_based_inference(model, dataloader, prototypes, threshold=0.8, device='cuda'):
    
    model.eval()
    total_samples, accepted_samples, rejected_samples, correct_predictions = 0, 0, 0, 0
    proto_tensor = torch.stack([prototypes[i] for i in range(len(prototypes))]).to(device)
    
    print(f"\n--- Metric Safe Rejection Activated (Threshold: {threshold}) ---")
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            embeddings, force = model(x)
            distances = torch.cdist(embeddings, proto_tensor, p=2.0)
            min_distances, predicted_classes = torch.min(distances, dim=1)
            
            for i in range(len(min_distances)):
                total_samples += 1
                dist = min_distances[i].item()
                pred_class = predicted_classes[i].item()
                true_class = y[i].item()
                
                if dist > threshold:
                    rejected_samples += 1
                else:
                    accepted_samples += 1
                    if pred_class == true_class: correct_predictions += 1
                        
    accuracy = (correct_predictions / accepted_samples) * 100 if accepted_samples > 0 else 0
    print("="*55)
    print("Metric Learning Performance & Safety Summary:")
    print(f"Total Tested: {total_samples} | Accepted: {accepted_samples} | Rejected: {rejected_samples}")
    print(f"Accuracy on Accepted: {accuracy:.2f}%")
    print("="*55)

def evaluate_metric_all_tasks(model, loader_t1, loader_t2, prototypes, threshold=0.8, device='cuda'):
    
    model.eval()
    all_preds, all_labels = [], []
    combined_dataset = ConcatDataset([loader_t1.dataset, loader_t2.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)
    proto_tensor = torch.stack([prototypes[i] for i in range(len(prototypes))]).to(device)
    
    with torch.no_grad():
        for x, y in combined_loader:
            x, y = x.to(device), y.to(device)
            embeddings, _ = model(x)
            distances = torch.cdist(embeddings, proto_tensor, p=2.0)
            min_distances, predicted_classes = torch.min(distances, dim=1)
            
            for i in range(len(min_distances)):
                dist = min_distances[i].item()
                true_class = y[i].item()
                pred_class = predicted_classes[i].item()
                all_labels.append(true_class)
                all_preds.append(6 if dist > threshold else pred_class) # 6 = Rejected

    class_names = ['Rest(0)', 'Mov1', 'Mov2', 'Mov3', 'Mov4', 'Mov5', 'Rejected']
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Unified Test Confusion Matrix (Threshold = {threshold})", fontsize=14, pad=15)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()
    
    print("\nClassification Report (Including Rejected Samples as class '6')")
    print(classification_report(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6], target_names=class_names, zero_division=0))

def test_doa_vs_baseline(baseline_model, doa_model, slda, target_loader, device='cuda'):
    
    baseline_model.eval()
    doa_model.eval()
    baseline_correct, doa_correct, total = 0, 0, 0
    
    print(" Testing Baseline Model vs DOA Model on Shifted (Target) Data...")
    with torch.no_grad():
        for x, y in target_loader:
            x, y = x.to(device), y.to(device)
            
            feat_base, _ = baseline_model(x)
            preds_base, _ = slda.predict(feat_base)
            baseline_correct += (preds_base == y).sum().item()
            
            feat_doa, _ = doa_model(x)
            preds_doa, _ = slda.predict(feat_doa)
            doa_correct += (preds_doa == y).sum().item()
            total += y.size(0)
            
    base_acc = (baseline_correct / total) * 100
    doa_acc = (doa_correct / total) * 100
    
    print("\n" + "="*60)
    print(" Domain Adaptation (DOA) Impact Report")
    print("="*60)
    print(f" Baseline Accuracy (Without DOA): {base_acc:.2f}%")
    print(f" Proposed DOA Accuracy (With CORAL): {doa_acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    
    print(" Evaluation Module Ready! Call the functions above to generate reports and plots.")