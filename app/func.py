from timeit import default_timer as timer
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import device, BATCH_SIZE


def train(train_dataloader, 
          valid_dataloader,
          model, 
          loss_fn, 
          accuracy_fn, 
          optimizer,  
          epochs: int, 
          scheduler = None):
    MODEL_PATH = Path(model.__class__.__name__)
    if not MODEL_PATH.exists():
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    start_timer = timer()
    total_train_loss = []
    total_train_accuracy = []
    total_valid_loss = []
    total_valid_accuracy = []
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(train_dataloader,
                                                model, 
                                                loss_fn, 
                                                accuracy_fn, 
                                                optimizer)
        total_train_loss.append(train_loss)
        total_train_accuracy.append(train_accuracy)
        
        valid_loss, valid_accuracy = valid_step(valid_dataloader,
                                                model, 
                                                loss_fn, 
                                                accuracy_fn)
        total_valid_loss.append(valid_loss)
        total_valid_accuracy.append(valid_accuracy)
        print(f"Epoch: {epoch+1}/{epochs} | ",
              f"Train loss: {train_loss:.4f} | ",
              f"Train accuracy: {train_accuracy:.2f} | ",
              f"Valid loss: {valid_loss:.4f} | ",
              f"Valid accuracy: {valid_accuracy:.2f}")
        if scheduler:
            scheduler.step(valid_loss)
        
        epoch_file = MODEL_PATH / f"{model.__class__.__name__}_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_file)

    end_timer = timer()
    time = end_timer - start_timer
    print(f"Training time: {time:.2f} seconds")
    show_results(total_train_accuracy,
                 total_train_loss,
                 total_valid_accuracy,
                 total_valid_loss,
                 epochs,
                 time,
                 model.__class__.__name__)
    
def train_step(train_dataloader, 
               model, 
               loss_fn, 
               accuracy_fn, 
               optimizer, 
               ) -> tuple[float, float]:
    """
    Returns:
        tuple[float, float]: total loss, total accuracy
    """
    train_total_loss = 0
    train_total_accuracy = 0
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_total_loss += loss.item()
        accuracy = accuracy_fn.to(device)(y, y_pred.argmax(dim=1))
        train_total_accuracy += accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"Batch: {batch+1}/{len(train_dataloader)} | ",
        #       f"Train loss: {loss.item():.4f} | ",
        #       f"Train accuracy: {accuracy:.2f}")
        
    train_total_loss /= len(train_dataloader)
    train_total_accuracy /= len(train_dataloader)
    return train_total_loss, train_total_accuracy

def valid_step(valid_dataloader, 
              model, 
              loss_fn, 
              accuracy_fn) -> tuple[float, float]:
    """
    Returns:
        tuple[float, float]: total loss, total accuracy
    """
    valid_total_loss = 0
    valid_total_accuracy = 0
    model.eval()
    with torch.inference_mode():
        for X, y in valid_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            valid_total_loss += loss.item()
            accuracy = accuracy_fn.to(device)(y, y_pred.argmax(dim=1))
            valid_total_accuracy += accuracy
        valid_total_loss /= len(valid_dataloader)
        valid_total_accuracy /= len(valid_dataloader)
    return valid_total_loss, valid_total_accuracy

def show_results(train_accuracy: list[float],
                 train_loss: list[float],
                 valid_accuracy: list[float],
                 valid_loss: list[float],
                 epochs: int,
                 time: float,
                 model_name: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f"{model_name} | {time:.2f} sec", fontsize=16)
    axs[0].plot(range(1, epochs + 1), train_accuracy, label='Train Accuracy', color='red')
    axs[0].plot(range(1, epochs + 1), valid_accuracy, label='Valid Accuracy', color='blue')
    axs[0].set_title(f"Train and Valid Accuracy")
    axs[0].legend()
    
    axs[1].plot(range(1, epochs + 1), train_loss, label='Train Loss', color='red')
    axs[1].plot(range(1, epochs + 1), valid_loss, label='Valid Loss', color='blue')
    axs[1].set_title(f"Train and Valid Loss")
    axs[1].legend()
    plt.show()
    
# рахує середнє і стандартне відхилення
def compute_mean_std(data_path, image_size: tuple[int, int]):
    transform = v2.Compose([
        v2.Resize(image_size),
        v2.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()
