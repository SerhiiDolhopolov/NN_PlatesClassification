from PIL import Image
import torch
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassStatScores

from config import device


def image_predict(image: str, augmentation_transform: type, model: torch.nn.Module) -> tuple[int, float]:
    """
    Predicts the label and probability of the image using the model.
    Args:
        image (str): Path to the image.
        model (torch.nn.Module): The trained model.
    Returns:
        tuple[int, float]: (label, probability)
    """
    img = Image.open(image).convert("RGB")
    img = augmentation_transform().get_test()(img).to(device)
    return predict(img, model)

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

def __compute_metrics(metric_type: type[MulticlassStatScores], y_true, y_pred, num_classes: int) -> dict:
    metric = metric_type(num_classes=num_classes)
    return metric(torch.tensor(y_pred), torch.tensor(y_true)).item()

def compute_precision(y_true, y_pred, num_classes: int) -> dict:
    return __compute_metrics(MulticlassPrecision, y_true, y_pred, num_classes)
  
def compute_recall(y_true, y_pred, num_classes: int) -> dict:
    return __compute_metrics(MulticlassRecall, y_true, y_pred, num_classes)

def compute_f1(y_true, y_pred, num_classes: int) -> dict:
    return __compute_metrics(MulticlassF1Score, y_true, y_pred, num_classes)