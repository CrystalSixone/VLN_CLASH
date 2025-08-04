import os, sys
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel, DPTForDepthEstimation, Dinov2Model

import pickle as pkl

class Dinov2RGBEncoder(nn.Module):
    def __init__(self, model_name, device="cuda", trainable=False):
        super().__init__()
        self.model_name = model_name
        self.trainable = trainable
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # self.model = Dinov2WithRegistersModel.from_pretrained(model_name).to(device)
        self.model = Dinov2Model.from_pretrained(model_name).to(device)

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, x, return_cls_features=True, need_process=False):
        if need_process:
            # during training, process in dataLoader
            x = self.processor(images=x, return_tensors="pt").to(self.device)
            x = self.model(**x)
            if return_cls_features:
                return x.pooler_output
            else:
                return x.last_hidden_state
        else:
            if len(x.shape) == 5: # [bs, 12, 3, 224, 224]
                img_nums = x.shape[1]
                bs = x.shape[0]
                x = x.reshape(-1, *x.shape[2:]) # 合并前两维 [bs*12, 3, 224, 224]
                x = self.model(x)
                if return_cls_features:
                    x = x.pooler_output
                else:
                    x = x.last_hidden_state
                x = x.reshape(bs, img_nums, -1) # 还原维度 [bs, 12, dim]
            else:
                x = self.model(x)
                if return_cls_features:
                    x = x.pooler_output
                else:
                    x = x.last_hidden_state
            return x # [bs, 12, dim]

class Dinov2DepthEncoder(nn.Module):
    def __init__(self, model_name, device="cuda", trainable=False):
        super().__init__()
        self.model_name = model_name
        self.trainable = trainable
        self.device = device
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(device)
        
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x, need_process=False):
        if need_process:
            x = self.processor(images=x, return_tensors="pt").to(self.device)
            x = self.model(**x, output_hidden_states=True)
            return x.hidden_states[-1][-1][0]
        else:
            if len(x.shape) == 5:  # [bs, 12, 3, 224, 224]
                img_nums = x.shape[1]
                bs = x.shape[0]
                x = x.reshape(-1, *x.shape[2:])  # [bs*12, 3, 224, 224]
                
                # 添加批处理逻辑, 否则太大的批次会报错：upsample_bilinear2d_nhwc only supports output tensors with less than INT_MAX elements
                batch_size = 12*48 # 12*16=192

                if x.shape[0] > batch_size:
                    # ATTENTION: This could lead very slow speed.
                    outputs = []
                    for i in range(0, x.shape[0], batch_size):
                        batch = x[i:i + batch_size]
                    output = self.model(batch, output_hidden_states=True)
                    outputs.append(output.hidden_states[-1])
                
                    x = torch.cat(outputs, dim=0)  # 连接所有批次的输出
                else:
                    x = self.model(x, output_hidden_states=True)
                    x = x.hidden_states[-1]

                x = x.reshape(bs, img_nums, *x.shape[1:])
                x = x[:,:,0]  # [bs, 12, dim]
            else:
                x = self.model(x, output_hidden_states=True)
                x = x.hidden_states[-1]
            return x

    def predicted_depth(self, x):
        # 同样添加批处理逻辑
        if isinstance(x, (list, tuple)):
            x = [self.processor(images=img, return_tensors="pt").to(self.device)["pixel_values"] for img in x]
            x = torch.cat(x, dim=0)
        else:
            x = self.processor(images=x, return_tensors="pt").to(self.device)["pixel_values"]
        
        batch_size = 32  # 可以根据您的GPU内存调整这个值
        depths = []
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i + batch_size]
            with torch.cuda.amp.autocast():
                output = self.model(batch)
                depths.append(output.predicted_depth)
        
        return torch.cat(depths, dim=0)

if __name__ == "__main__":
    rgb_encoder = Dinov2RGBEncoder(model_name="data/pretrained/dinov2-large")
    depth_encoder = Dinov2DepthEncoder(model_name="data/pretrained/dpt-dinov2-large-nyu")
    
    image_info_file = "/share/home/tj90055/w61/habitat/waypoint-predictor-main/training_data/rgbd_fov90_hm3d_include_downViews/hm3d/train/00000-kfPV7w3FaU5/00000-kfPV7w3FaU5_00000_hm3d_imgs.pkl"
    
    rgb_depth_img = pkl.load(open(image_info_file, "rb"))
    rgb_img = rgb_depth_img['rgb']
    depth_img = rgb_depth_img['depth']
    
    rgb_img = Image.fromarray(rgb_img[0][:,:,:3])
    depth_img = Image.fromarray(depth_img[0].squeeze(), mode='L')
    
    with torch.no_grad():
        rgb_features = rgb_encoder(rgb_img, return_cls_features=True)
        depth_features = depth_encoder(rgb_img)
        depth_features = depth_encoder.predicted_depth(rgb_img)
    
    # 打印特征的形状
    print("RGB features shape:", rgb_features.shape)
    print("Depth features shape:", depth_features.shape)
    
    # 可视化预测的depth
    output = depth_features.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save("data/debug/predicted_depth.jpg")
    
    '''比较预测depth和真值depth'''
    # 将预测的深度图转换为numpy数组
    pred_depth = depth_features.cpu().numpy().squeeze()
    
    # 将真实深度图转换为numpy数组
    true_depth = np.array(depth_img)
    
    # 将预测深度图的尺寸调整为与真实深度图相同
    from PIL import Image
    pred_depth_resized = Image.fromarray(pred_depth).resize(true_depth.shape[::-1], Image.BILINEAR)
    pred_depth = np.array(pred_depth_resized)
    
    # 归一化两个深度图到相同的范围 [0,1]
    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
    true_depth = (true_depth - true_depth.min()) / (true_depth.max() - true_depth.min())
    
    # 计算误差
    mse = np.mean((pred_depth - true_depth) ** 2)
    mae = np.mean(np.abs(pred_depth - true_depth))
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # 可视化对比
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(true_depth, cmap='viridis')
    plt.title('Ground Truth Depth')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(pred_depth, cmap='viridis')
    plt.title('Predicted Depth')
    plt.colorbar()
    
    plt.subplot(133)
    diff = np.abs(pred_depth - true_depth)
    plt.imshow(diff, cmap='hot')
    plt.title('Absolute Difference')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("data/debug/depth_comparison.png")
    plt.close()