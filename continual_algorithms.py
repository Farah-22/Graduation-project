# continual_algorithms.py
import torch
import torch.nn.functional as F

class StreamingLDA:
    def __init__(self, input_dim=32, shrinkage_param=1e-4, device='cuda'):
        """
        Streaming Linear Discriminant Analysis (SLDA).
        Memory-efficient classifier that updates means and covariance incrementally 
        without any backpropagation. Ideal for Edge AI.
        """
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
            
            # 1. Update scatter matrix (Covariance calculation)
            centered_features = class_features - batch_mean
            self.scatter_matrix += torch.matmul(centered_features.T, centered_features)
            
            # 2. Streaming update for the class mean
            self.class_means[c] = (self.class_means[c] * old_count + batch_mean * new_count) / total_count
            self.class_counts[c] = total_count
            self.total_samples += new_count
            
        self._compute_precision_matrix()

    def _compute_precision_matrix(self):
        k = len(self.class_means)
        if k == 0 or self.total_samples <= k:
            return
            
        # Normalize scatter to get actual covariance
        covariance = self.scatter_matrix / (self.total_samples - k)
            
        # Apply Shrinkage to ensure the matrix is invertible (Numerical stability)
        identity = torch.eye(self.input_dim).to(self.device)
        cov_shrunk = (1 - self.shrinkage_param) * covariance + self.shrinkage_param * identity
        
        # Invert to get precision matrix
        self.precision_matrix = torch.linalg.inv(cov_shrunk)

    def fit_base(self, dataloader, backbone):
        backbone.eval()
        print("\n [Deep SLDA] Fitting initial Covariance Matrix on Base Classes...")
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features, _ = backbone(inputs)
                self._update_scatter(features, labels)
        print(f" [Deep SLDA] Fit complete. Tracking {len(self.class_means)} base classes.")

    def stream_new_data(self, dataloader, backbone):
        backbone.eval()
        print("\n [Deep SLDA] Streaming Few-Shot update for Novel Classes...")
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features, _ = backbone(inputs)
                self._update_scatter(features, labels)
        print(f" [Deep SLDA] Update complete. Total tracked classes: {len(self.class_means)}.")

    def predict(self, features):
        if self.precision_matrix is None:
            raise ValueError("Model not fitted yet.")
            
        classes = list(self.class_means.keys())
        M = torch.stack([self.class_means[c] for c in classes])
        W = torch.matmul(self.precision_matrix, M.T)
        b = -0.5 * torch.sum(M * torch.matmul(M, self.precision_matrix), dim=1)
        scores = torch.matmul(features, W) + b
        
        max_scores, preds_idx = torch.max(scores, dim=1)
        predicted_classes = torch.tensor([classes[i] for i in preds_idx]).to(self.device)
        
        return predicted_classes, max_scores


def compute_class_prototypes(model, dataloader, num_classes=3, device='cuda'):
    """
    Calculates the prototype (mean embedding) for each class.
    This acts as the 'center' for each movement in the metric space.
    """
    model.eval()
    class_embeddings = {i: [] for i in range(num_classes)}
    
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
            
    return prototypes


def coral_loss(source_features, target_features):
    """
    Deep CORAL Loss: Aligns the second-order statistics (covariances) 
    of the Source domain and Target domain.
    """
    d = source_features.size(1)
    
    source_mean = torch.mean(source_features, 0, keepdim=True)
    source_centered = source_features - source_mean
    source_cov = (source_centered.t() @ source_centered) / (source_features.size(0) - 1)
    
    target_mean = torch.mean(target_features, 0, keepdim=True)
    target_centered = target_features - target_mean
    target_cov = (target_centered.t() @ target_centered) / (target_features.size(0) - 1)
    
    loss = torch.sum(torch.pow(source_cov - target_cov, 2))
    loss = loss / (4 * d * d)
    
    return loss