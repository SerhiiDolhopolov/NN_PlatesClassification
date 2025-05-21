import torch
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassStatScores

from config import device


def predict(X, model: torch.nn.Module) -> tuple[int, float]:
    """
    Returns:
        tuple[int, float]: (label, probability)
    """
    model.eval()
    with torch.inference_mode():
        X = torch.unsqueeze(X, dim=0).to(device)
        y_pred = model(X)
        prob = torch.softmax(y_pred, dim=1)
        max_prob, predict_label = torch.max(prob, dim=1)
    return predict_label.item(), max_prob.item()

def compute_metrics(metric_type: type[MulticlassStatScores], y_true, y_pred, num_classes: int) -> dict:
    metric = metric_type(num_classes=num_classes)
    return metric(torch.tensor(y_pred), torch.tensor(y_true)).item()

def compute_precision(y_true, y_pred, num_classes: int) -> dict:
    return compute_metrics(MulticlassPrecision, y_true, y_pred, num_classes)
  
def compute_recall(y_true, y_pred, num_classes: int) -> dict:
    return compute_metrics(MulticlassRecall, y_true, y_pred, num_classes)

def compute_f1(y_true, y_pred, num_classes: int) -> dict:
    return compute_metrics(MulticlassF1Score, y_true, y_pred, num_classes)