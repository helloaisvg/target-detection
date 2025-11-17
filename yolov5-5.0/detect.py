# YOLOv5 detection script
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from models.yolo import Model
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
import yaml

def detect(opt):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    with open(opt.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model = Model(cfg, ch=3, nc=cfg['nc']).to(device)
    
    # Load weights
    if opt.weights:
        weights_path = Path(opt.weights)
        if not weights_path.exists():
            # Try to find in runs/train/exp*/weights/
            runs_dir = Path('runs/train')
            if runs_dir.exists():
                exp_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('exp')], 
                                key=lambda x: int(x.name[3:]) if x.name[3:].isdigit() else -1, reverse=True)
                for exp_dir in exp_dirs:
                    weights_dir = exp_dir / 'weights'
                    if weights_dir.exists():
                        # Try to find best.pt or last.pt
                        for weight_file in ['best.pt', 'last.pt', 'coco128.pt']:
                            candidate = weights_dir / weight_file
                            if candidate.exists():
                                weights_path = candidate
                                print(f'Found weights in {weights_path}')
                                break
                        if weights_path.exists():
                            break
        
        if weights_path.exists():
            ckpt = torch.load(str(weights_path), map_location=device)
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
                else:
                    state_dict = ckpt
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            print(f'Loaded weights from {weights_path}')
        else:
            print(f'Warning: Weights file {opt.weights} not found, using random initialization')
    else:
        print('Warning: No weights specified, using random initialization')
    
    model.eval()
    
    # Load image
    source = Path(opt.source)
    if source.is_file():
        img_files = [source]
    elif source.is_dir():
        img_files = sorted(list(source.glob('*.jpg')) + list(source.glob('*.png')))
    else:
        raise ValueError(f'Invalid source: {opt.source}')
    
    # Process images
    for img_path in img_files:
        # Read image
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            print(f'Failed to load image: {img_path}')
            continue
        
        # Preprocess
        img = letterbox(img0, new_shape=opt.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = model(img, augment=opt.augment)
            # Handle different output formats
            # Model returns (torch.cat(z, 1), x) in eval mode, or list in training mode
            if isinstance(output, tuple):
                # Eval mode: (concatenated_output, raw_outputs)
                pred = output[0]  # Use concatenated output
            elif isinstance(output, list):
                # Training mode or list of outputs
                if len(output) > 0:
                    pred = torch.cat(output, 1) if len(output) > 1 else output[0]
                else:
                    pred = output[0]
            else:
                pred = output
        
        # Apply NMS with higher IoU threshold to reduce overlapping boxes
        pred = non_max_suppression(pred, opt.conf_thres, iou_thres=0.5, classes=opt.classes, agnostic=opt.agnostic_nms)
        
        # Process detections
        for i, det in enumerate(pred):
            if len(det):
                # Filter valid detections (confidence and coordinates)
                h0, w0 = img0.shape[:2]
                
                # Rescale boxes from img_size to img0 size first
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                # Clip coordinates to image bounds first
                det[:, 0].clamp_(0, w0)
                det[:, 1].clamp_(0, h0)
                det[:, 2].clamp_(0, w0)
                det[:, 3].clamp_(0, h0)
                
                # Filter valid detections with less strict criteria
                valid = (det[:, 4] > opt.conf_thres) & \
                        (det[:, 0] < det[:, 2]) & (det[:, 1] < det[:, 3]) & \
                        ((det[:, 2] - det[:, 0]) > 5) & ((det[:, 3] - det[:, 1]) > 5)  # min width and height reduced to 5
                
                det = det[valid]
                
                if len(det) > 0:
                    # Sort by confidence (highest first) and limit number of detections
                    det = det[det[:, 4].argsort(descending=True)[:100]]  # Keep top 100 detections
                    
                    # Draw boxes
                    class_names = cfg.get('names', [f'class_{i}' for i in range(cfg.get('nc', 80))])
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        # Ensure valid box with minimum size
                        if (x2 - x1) > 5 and (y2 - y1) > 5:
                            cls_id = int(cls)
                            if cls_id < len(class_names):
                                label = f'{class_names[cls_id]} {conf:.2f}'
                            else:
                                label = f'class_{cls_id} {conf:.2f}'
                            plot_one_box([x1, y1, x2, y2], img0, label=label, color=(0, 255, 0), line_thickness=2)
        
        # Save result to runs/detect/exp{n}/
        save_dir = Path('runs/detect')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Find next available exp directory
        exp_num = 0
        while (save_dir / f'exp{exp_num}').exists():
            exp_num += 1
        exp_dir = save_dir / f'exp{exp_num}'
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = exp_dir / img_path.name
        cv2.imwrite(str(save_path), img0)
        print(f'Result saved to {save_path}')

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (ratio, (dw, dh))

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """绘制检测框和标签"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    # 绘制检测框（使用更粗的线条使其更明显）
    cv2.rectangle(img, c1, c2, color, thickness=max(tl, 2), lineType=cv2.LINE_AA)
    
    if label:
        # 绘制标签背景和文字
        tf = max(tl - 1, 1)
        font_scale = max(0.5, tl / 3)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, tf)
        
        # 标签背景框
        label_y = max(c1[1], text_height + 5)
        cv2.rectangle(img, 
                     (c1[0], label_y - text_height - 5), 
                     (c1[0] + text_width + 5, label_y + baseline), 
                     color, -1, cv2.LINE_AA)
        
        # 标签文字
        cv2.putText(img, label, (c1[0] + 2, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/images', help='source')
    parser.add_argument('--weights', type=str, default='./weights/coco128.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    
    opt = parser.parse_args()
    detect(opt)

