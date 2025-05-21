import os

import torch


NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32
SEED = 1

TRAIN_DATA_PATH = "../data/train"
TEST_DATA_PATH = "../data/test"
FOLDERS_DATA_TYPE = [TRAIN_DATA_PATH, TEST_DATA_PATH]
FOLDERS_CLASS_TYPE = ['clean', 'dirty']

device = "cuda" if torch.cuda.is_available() else "cpu"
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)