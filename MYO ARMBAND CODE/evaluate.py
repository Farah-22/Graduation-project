# evaluate.py
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_inference(model, slda, dataloader, num_passes=5, uncertainty_threshold=0.2, device='cuda'):
    model.eval()
    enable_dropout(model)
    
    total_samples = 0
    rejected_samples = 0
    accepted_correct = 0
    
    class_total = {cls: 0 for cls in range(6)}
    class_rejected = {cls: 0 for cls in range(6)}
    class_accepted_correct = {cls: 0 for cls in range(6)}
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            
            pass_predictions = torch.zeros(num_passes, batch_size, dtype=torch.long).to(device)
            
            for p in range(num_passes):
                features, _ = model(inputs)
                preds, _ = slda.predict(features)
                pass_predictions[p] = preds
                
            for i in range(batch_size):
                total_samples += 1
                true_label = labels[i].item()
                class_total[true_label] += 1
                
                sample_preds = pass_predictions[:, i]
                uncertainty = torch.var(sample_preds.float()).item()
                final_pred = torch.mode(sample_preds).values.item()
                
                if uncertainty > uncertainty_threshold:
                    rejected_samples += 1
                    class_rejected[true_label] += 1
                else:
                    if final_pred == true_label:
                        accepted_correct += 1
                        class_accepted_correct[true_label] += 1
                        
    accepted_samples = total_samples - rejected_samples
    accuracy = (accepted_correct / accepted_samples) * 100 if accepted_samples > 0 else 0
    
    return {
        'total': total_samples,
        'rejected': rejected_samples,
        'accepted': accepted_samples,
        'accuracy': accuracy,
        'class_total': class_total,
        'class_rejected': class_rejected,
        'class_accepted_correct': class_accepted_correct
    }

def test_recalibrated_slda(backbone, original_slda, recalibrated_slda, 
                            target_loader, device='cuda'):
    backbone.eval()
    
    original_correct = 0
    recalibrated_correct = 0
    total = 0
    
    original_preds = []
    recalibrated_preds = []
    all_labels = []
    
    print("\n" + "="*70)
    print("Testing SLDA Original vs Recalibrated")
    print("="*70)
    
    with torch.no_grad():
        for x, y in target_loader:
            x, y = x.to(device), y.to(device)
            
            embeddings, _ = backbone(x)
            
            preds_orig, _ = original_slda.predict(embeddings)
            original_correct += (preds_orig == y).sum().item()
            
            preds_recal, _ = recalibrated_slda.predict(embeddings)
            recalibrated_correct += (preds_recal == y).sum().item()
            
            original_preds.extend(preds_orig.cpu().numpy())
            recalibrated_preds.extend(preds_recal.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            total += y.size(0)
    
    orig_acc = (original_correct / total) * 100
    recal_acc = (recalibrated_correct / total) * 100
    
    print(f"\nResults:")
    print(f"   Original SLDA (without recalibration): {orig_acc:.2f}%")
    print(f"   Recalibrated SLDA (with recalibration): {recal_acc:.2f}%")
    print(f"   Improvement: {recal_acc - orig_acc:+.2f}%")
    
    if recal_acc > orig_acc:
        print(f"\nSLDA Re-calibration improved accuracy by {recal_acc - orig_acc:.2f}%!")
    else:
        print(f"\nRe-calibration did not improve accuracy.")
    
    print("\nPer-Class Analysis:")
    print(f"{'Class':<10} {'Original SLDA':<15} {'Recalibrated':<15} {'Improvement':<12}")
    print("-" * 55)
    
    for cls in range(3):
        cls_mask = [i for i, lbl in enumerate(all_labels) if lbl == cls]
        if len(cls_mask) > 0:
            orig_cls_correct = sum(1 for i in cls_mask if original_preds[i] == cls)
            recal_cls_correct = sum(1 for i in cls_mask if recalibrated_preds[i] == cls)
            orig_cls_acc = (orig_cls_correct / len(cls_mask)) * 100
            recal_cls_acc = (recal_cls_correct / len(cls_mask)) * 100
            improvement = recal_cls_acc - orig_cls_acc
            print(f"Class {cls:<5} {orig_cls_acc:<15.2f} {recal_cls_acc:<15.2f} {improvement:<+11.2f}%")
    
    class_names = ['Rest(0)', 'Mov1', 'Mov2']
    cm = confusion_matrix(all_labels, recalibrated_preds, labels=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Recalibrated SLDA on Shifted Data (Accuracy: {recal_acc:.2f}%)", 
              fontsize=14, pad=15)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()