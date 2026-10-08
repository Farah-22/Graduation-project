import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels, class_names, title, cmap='Blues', figsize=(14, 10)):
    """
    Plots a confusion matrix using Seaborn.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of unique class labels for axis ordering.
        class_names: List of string names for each class.
        title: Title of the plot.
        cmap: Color map for the heatmap.
        figsize: Size of the figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()