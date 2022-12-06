
import os
import time
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
from utils import *
from dataset import *
from model import build_unet

def train(model, loader, optimizer, loss_fn, device):
    print("Train...")
    epoch_loss = 0.0

    
    model.train()
    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device, threshold = 0.6):
    print("Valid...")
    epoch_loss = 0.0

    iou_arr = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            threshold, iou = iou_pytorch(y_pred, y)

            iou_arr.append(iou.cpu().numpy())
        mean_iou = np.mean(iou_arr)
        print('Mean IoU: {:.4f}'.format(mean_iou))
        epoch_loss = epoch_loss/len(loader)
        
    return epoch_loss


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    '''
    Code Unet for ----------  2 CLASSES ----------
    '''

    seeding(1234)
    """ Load dataset """
    train_x = sorted(glob("./dataset_full/dataset_full/train/*"))[:50]
    train_y = sorted(glob("./dataset_full/dataset_full/trainanot_full/*"))[:50]
    valid_x = sorted(glob("./dataset_full/dataset_full/val/*"))[:50]
    valid_y = sorted(glob("./dataset_full/dataset_full/valanot_full/*"))[:50]

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 80
    W = 160
    size = (W, H)
    batch_size = 1
    num_epochs = 30
    lr = 1e-4
    checkpoint_path = "./best.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y, W, H)
    valid_dataset = DriveDataset(valid_x, valid_y, W, H)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    device = torch.device('cuda')   ## GTX 1060 6GB
    model = build_unet()
    model = model.to(device)

    torch.backends.cudnn.benchmark = True # new
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    loss_train_plt = []
    loss_val_plt = []

    for epoch in range(num_epochs):
        print("Epoch:", epoch+1)
        train_loss = train(model, train_loader, optimizer, loss_fn, device)    
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        
        """ Saving the model """
        if valid_loss < best_valid_loss:
            print(f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}")

            best_valid_loss = valid_loss
            name = os.path.join(os.getcwd(), "output_" + str(epoch) + ".pth")
            torch.save(model.state_dict(), name)

        print(f'\tTrain Loss: {train_loss:.3f}\n\t Val. Loss: {valid_loss:.3f}\n')
        
        loss_train_plt.append(train_loss)
        loss_val_plt.append(valid_loss)
        plt.plot(loss_train_plt)
        plt.plot(loss_val_plt)
        plt.savefig(f"loss_epoch_{epoch}.png")  