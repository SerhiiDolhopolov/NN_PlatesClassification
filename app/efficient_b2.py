from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torch

from config import device, FOLDERS_CLASS_TYPE



def get_model(state_dict_path : str = None) -> torch.nn.Module:
    efficientnet_b2_model = efficientnet_b2(weights=EfficientNet_B2_Weights).to(device)
    
    for param in efficientnet_b2_model.parameters():
        param.requires_grad = False
        
    efficientnet_b2_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=False),
        torch.nn.Linear(in_features=1408,
                        out_features=256,
                        bias=True),
        torch.nn.ReLU(inplace=False),
        torch.nn.Dropout(0.5, inplace=False),
        torch.nn.Linear(in_features=256,
                        out_features=len(FOLDERS_CLASS_TYPE),
                        bias=True)).to(device)
    if state_dict_path:
        efficientnet_b2_model.load_state_dict(torch.load(state_dict_path))
    return efficientnet_b2_model
