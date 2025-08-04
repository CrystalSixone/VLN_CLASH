import torch
import torch.nn as nn
import numpy as np
import utils
import time

from transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig

from src.trainer.models.predictor_network import WaypointPredictor, WP_predict
from src.trainer.models.Dinov2_encoders import Dinov2RGBEncoder, Dinov2DepthEncoder
from src.trainer.models.Resnet_encoders import DepthEncoder
from src.trainer.models.CLIPH14_encoders import CLIPH14Encoder

class WP_Dinov2_net(nn.Module):
    def __init__(self, args, logger):
        super(WP_Dinov2_net, self).__init__()
        self.args = args
        self.logger = logger
        
        # RGB Encoder
        if self.args.rgb_encoder_type == 'dinov2':
            self.rgb_encoder = Dinov2RGBEncoder(model_name="data/pretrained/dinov2-large")
        elif self.args.rgb_encoder_type == 'cliph14':
            self.rgb_encoder = CLIPH14Encoder(model_name="data/pretrained/CLIP-H14/open_clip_pytorch_model.bin")

        # Depth Encoder 
        if self.args.depth_encoder_type == 'dinov2':
            self.depth_encoder = Dinov2DepthEncoder(model_name="data/pretrained/dpt-dinov2-large-nyu")
        elif self.args.depth_encoder_type == 'resnet':
            self.depth_encoder = DepthEncoder()
        input_dim = 1024 # 1024 for dinov2-large
        self.wp_predictor = WaypointPredictor(args=args,
            rgb_input_dim=input_dim,
            depth_input_dim=input_dim,
            hidden_dim=args.HIDDEN_DIM, n_classes=args.NUM_CLASSES)

        # freeze the encoder parameters
        for param in self.rgb_encoder.parameters():
            param.requires_grad = False
        for param in self.depth_encoder.parameters():
            param.requires_grad = False
        
        # unfreeze the predictor parameters
        for param in self.wp_predictor.parameters():
            param.requires_grad = True

    def forward(self, batch):
        mode = batch['mode']

        if mode in ['train', 'eval']:
            if not self.args.load_from_lmdb:
                # process the features
                with torch.no_grad():
                    rgb_feats = self.rgb_encoder(batch['rgb'])
                    depth_feats = self.depth_encoder(batch['depth'])
                    if self.args.include_down_views:
                        rgb_down_feats = self.rgb_encoder(batch['rgb_down'])
                        depth_down_feats = self.depth_encoder(batch['depth_down'])
                    else:
                        rgb_down_feats = None
                        depth_down_feats = None
            else:
                rgb_feats = batch['rgb']
                depth_feats = batch['depth']
                if self.args.include_down_views:
                    rgb_down_feats = batch['rgb_down']
                    depth_down_feats = batch['depth_down']
                else:
                    rgb_down_feats = None
                    depth_down_feats = None
                
                # if self.args.depth_encoder_type == 'resnet':
                #     depth_feats = self.depth_encoder(depth_feats)
                #     if self.args.include_down_views:
                #         depth_down_feats = self.depth_encoder(depth_down_feats)

            vis_logits = self.wp_predictor(rgb_feats, depth_feats, rgb_down_feats, depth_down_feats)
            # entry-wise probabilities
            if self.args.train_use_sigmoid:
                vis_logits = torch.sigmoid(vis_logits)
                vis_probs = vis_logits
            else:
                # Original
                vis_probs = torch.sigmoid(vis_logits) # TODO：为什么训练的时候不加sigmoid?

            if mode == 'train':
                return vis_logits
            elif mode == 'eval':
                return vis_probs, vis_logits

        elif mode == 'pre_extract_features':
            with torch.no_grad():
                rgb_feats = self.rgb_encoder(batch['rgb'])
                depth_feats = self.depth_encoder(batch['depth'])
                if self.args.depth_encoder_type == 'resnet':
                    depth_feats = depth_feats.reshape(rgb_feats.shape[0], rgb_feats.shape[1], *depth_feats.shape[1:])
                if self.args.include_down_views:
                    rgb_down_feats = self.rgb_encoder(batch['rgb_down'])
                    depth_down_feats = self.depth_encoder(batch['depth_down'])
                    if self.args.depth_encoder_type == 'resnet':
                        depth_down_feats = depth_down_feats.reshape(rgb_down_feats.shape[0], rgb_feats.shape[1], *depth_down_feats.shape[1:])
                else:
                    rgb_down_feats = None
                    depth_down_feats = None
                return rgb_feats, depth_feats, rgb_down_feats, depth_down_feats

    # def forward(self, batch):
    #     mode = batch['mode']

    #     with torch.no_grad():
    #         if self.args.include_down_views:
    #             # 批量处理所有图像
    #             combined_rgb = torch.cat([batch['rgb'], batch['rgb_down']], dim=0)
    #             combined_depth = torch.cat([batch['depth'], batch['depth_down']], dim=0)
                
    #             if self.args.analysis_time:
    #                 start_time = time.time()
    #             # 一次性提取所有特征
    #             all_rgb_feats = self.rgb_encoder(combined_rgb)
    #             if self.args.analysis_time:
    #                 print(f"All RGB encoder time: {time.time() - start_time}")
                
    #             if self.args.analysis_time:
    #                 start_time = time.time()
    #             all_depth_feats = self.depth_encoder(combined_depth)
    #             if self.args.analysis_time:
    #                 print(f"All depth encoder time: {time.time() - start_time}")
                
    #             # 分离特征
    #             batch_size = batch['rgb'].shape[0]
    #             rgb_feats = all_rgb_feats[:batch_size]
    #             rgb_down_feats = all_rgb_feats[batch_size:]
    #             depth_feats = all_depth_feats[:batch_size]
    #             depth_down_feats = all_depth_feats[batch_size:]
    #         else:
    #             if self.args.analysis_time:
    #                 start_time = time.time()
    #             rgb_feats = self.rgb_encoder(batch['rgb'])
    #             if self.args.analysis_time:
    #                 print(f"RGB encoder time: {time.time() - start_time}")
    #             if self.args.analysis_time:
    #                 start_time = time.time()
    #             depth_feats = self.depth_encoder(batch['depth'])
    #             if self.args.analysis_time:
    #                 print(f"Depth encoder time: {time.time() - start_time}")
    #             rgb_down_feats = None
    #             depth_down_feats = None
    #     if self.args.analysis_time:
    #         start_time = time.time()
    #     vis_logits = self.wp_predictor(rgb_feats, depth_feats, rgb_down_feats, depth_down_feats)
    #     if self.args.analysis_time:
    #         print(f"WP predictor time: {time.time() - start_time}")
    #     # entry-wise probabilities
    #     if self.args.train_use_sigmoid:
    #         vis_logits = torch.sigmoid(vis_logits)
    #         vis_probs = vis_logits
    #     else:
    #         # Original
    #         vis_probs = torch.sigmoid(vis_logits) # TODO：为什么训练的时候不加sigmoid?

    #     if mode == 'train':
    #         return vis_logits
    #     elif mode == 'eval':
    #         return vis_probs, vis_logits
        