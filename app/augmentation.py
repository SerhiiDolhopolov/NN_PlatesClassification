import torch
from torchvision.transforms import v2


class AugmentationV1:
    size = (128, 128)
    mean = [0.519, 0.499, 0.469]
    std = [0.175, 0.183, 0.197]
        
    def get_train(self):
        compose_list = [
            v2.Resize(self.size),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),          
            v2.RandomRotation(degrees=10),          
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            v2.TrivialAugmentWide(num_magnitude_bins=31),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
        ]        
        augment = v2.Compose(compose_list)
        return augment
        
    def get_test(self):
        compose_list = [
            v2.Resize(self.size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
        ]
        augment = v2.Compose(compose_list)
        return augment
    
#https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2
class AugmentationEfficientnet_b2:
    size = (288, 288)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    def get_train(self):
        compose_list = [
            v2.Resize(self.size),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),          
            v2.RandomRotation(degrees=10),          
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            v2.TrivialAugmentWide(num_magnitude_bins=31),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
        ]        
        augment = v2.Compose(compose_list)
        return augment
    
    def get_test(self):
        compose_list = [
            v2.Resize(self.size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
        ]
        augment = v2.Compose(compose_list)
        return augment