#!/usr/bin/env python3
# YOLOv5 training script
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.yolo import Model
from utils.datasets import LoadImagesAndLabels
from utils.general import labels_to_class_weights
import yaml
from pathlib import Path
import os

def train(hyp, opt, device, tb_writer=None):
    # Load model
    with open(opt.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model = Model(cfg, ch=3, nc=cfg['nc']).to(device)
    
    # Load pretrained weights if specified
    if opt.weights:
        if Path(opt.weights).exists():
            print(f'Loading pretrained weights from {opt.weights}')
            try:
                # Try loading with weights_only=False for compatibility
                ckpt = torch.load(opt.weights, map_location=device, weights_only=False)
                if isinstance(ckpt, dict):
                    # Handle different checkpoint formats
                    if 'model' in ckpt:
                        # Official YOLOv5 format
                        if hasattr(ckpt['model'], 'state_dict'):
                            state_dict = ckpt['model'].state_dict()
                        elif isinstance(ckpt['model'], dict):
                            state_dict = ckpt['model']
                        else:
                            state_dict = ckpt['model']
                        # Filter out incompatible keys
                        model_dict = model.state_dict()
                        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                        model.load_state_dict(filtered_dict, strict=False)
                        print(f'Loaded {len(filtered_dict)}/{len(state_dict)} compatible layers from pretrained weights')
                    else:
                        model.load_state_dict(ckpt, strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
                print('Pretrained weights loaded successfully')
            except Exception as e:
                print(f'Warning: Could not load pretrained weights: {e}')
                print('Training from scratch instead')
        else:
            print(f'Warning: Weights file {opt.weights} not found, training from scratch')
    
    # Load dataset
    dataset = LoadImagesAndLabels(
        opt.data,
        img_size=opt.img,
        batch_size=opt.batch,
        augment=True,
        hyp=hyp
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch,
        shuffle=True,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyp['lr0'],
        momentum=hyp['momentum'],
        weight_decay=hyp['weight_decay'],
        nesterov=True
    )
    
    # Scheduler
    lf = lambda x: (1 - x / opt.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # Class weights (simplified - would need actual labels in real implementation)
    model.nc = cfg['nc']
    model.class_weights = torch.ones(model.nc).to(device)
    
    # Training loop
    model.train()
    
    # Create a single exp directory for this training run
    save_dir = Path('runs/train')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next available exp directory for this training session
    exp_num = 0
    while (save_dir / f'exp{exp_num}').exists():
        exp_num += 1
    exp_dir = save_dir / f'exp{exp_num}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = exp_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Training results will be saved to: {exp_dir}')
    
    best_loss = float('inf')
    
    for epoch in range(opt.epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward
            pred = model(imgs)
            
            # Compute loss (simplified)
            loss = compute_loss(pred, targets, model)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_i % 10 == 0:
                print(f'Epoch {epoch}/{opt.epochs}, Batch {batch_i}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else epoch_loss
        scheduler.step()
        
        # Save checkpoint every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == opt.epochs - 1:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }
            # Save to the same exp directory
            torch.save(ckpt, weights_dir / f'epoch{epoch}.pt')
            print(f'Checkpoint saved: {weights_dir / f"epoch{epoch}.pt"}')
        
        # Update best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(best_ckpt, weights_dir / 'best.pt')
            print(f'Best model updated at epoch {epoch}, loss: {avg_loss:.4f}')
    
    # Save final model to the same exp directory
    final_ckpt = {
        'epoch': opt.epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': avg_loss,
    }
    torch.save(final_ckpt, weights_dir / 'last.pt')
    # best.pt is already saved during training
    
    # Also save to weights directory for easy access
    weights_main = Path('weights')
    weights_main.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_main / 'coco128.pt')
    
    print(f'Training completed.')
    print(f'Model saved to: {weights_dir / "last.pt"} and {weights_dir / "best.pt"}')
    print(f'Also saved to: {weights_main / "coco128.pt"}')
    print(f'To use for detection, copy best.pt to weights/coco128.pt or use: {weights_dir / "best.pt"}')

def compute_loss(pred, targets, model):
    """Simplified loss computation"""
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    
    # Simple MSE loss for demonstration
    # In real YOLOv5, this would be more complex (box, obj, cls losses)
    loss = nn.MSELoss()(pred, torch.zeros_like(pred))
    return loss

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    # Pad labels to same length
    max_labels = max(len(l) for l in labels)
    padded_labels = []
    for l in labels:
        if len(l) < max_labels:
            pad = torch.zeros((max_labels - len(l), 5))
            l = torch.cat([l, pad], dim=0)
        padded_labels.append(l)
    labels = torch.stack(padded_labels, 0)
    return imgs, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--data', type=str, default='./data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    opt = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    hyp = {
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
    }
    
    train(hyp, opt, device)

