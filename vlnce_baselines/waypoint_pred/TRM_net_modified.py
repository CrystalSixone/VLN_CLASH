import torch
import torch.nn as nn
import numpy as np
import vlnce_baselines.waypoint_pred.utils as utils

from .transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def TRM_predict(mode, args, predictor, rgb_feats, depth_feats, rgb_down_feats=None, depth_down_feats=None):
    ''' predicting the waypoint probabilities '''
    vis_logits = predictor(rgb_feats, depth_feats, rgb_down_feats, depth_down_feats)
    # entry-wise probabilities
    vis_probs = torch.sigmoid(vis_logits)

    if mode == 'train':
        return vis_logits
    elif mode == 'eval':
        return vis_probs, vis_logits


class BinaryDistPredictor_TRM_modified(nn.Module):
    def __init__(self, args=None, hidden_dim=768, n_classes=12):
        super(BinaryDistPredictor_TRM_modified, self).__init__()
        self.args = args

        self.num_angles = args.angles
        self.num_imgs = args.num_imgs
        self.n_classes = n_classes

        if self.args.encoder_type == 'resnet':
            self.visual_fc_rgb = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod([2048,7,7]), hidden_dim),
                nn.ReLU(True),
            )
        elif self.args.encoder_type == 'dinov2':
            self.visual_fc_rgb = nn.Sequential(
                nn.Linear(1024, hidden_dim),
                nn.ReLU(True),
            )
        self.visual_fc_depth = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([128,4,4]), hidden_dim),
            nn.ReLU(True),
        )
        self.visual_merge = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(True),
        )

        config = BertConfig()
        config.model_type = 'visual'
        config.finetuning_task = 'waypoint_predictor'
        config.hidden_dropout_prob = 0.3
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = args.TRM_LAYER
        self.waypoint_TRM = WaypointBert(config=config)

        layer_norm_eps = config.layer_norm_eps

        self.mask = utils.get_attention_mask(
            num_imgs=self.num_imgs,
            neighbor=args.TRM_NEIGHBOR).to(device)

        self.vis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                int(n_classes*(self.num_angles/self.num_imgs))),
        )
        
        self.rgb_dropout = nn.Dropout(self.args.rgb_dropout_prob)
        self.depth_dropout = nn.Dropout(self.args.depth_dropout_prob)
        self.visual_dropout = nn.Dropout(self.args.visual_dropout_prob)
        
    def forward(self, rgb_feats, depth_feats):
        if len(rgb_feats.shape) == 2:
            bsi = rgb_feats.shape[0] // self.num_imgs
        else:
            bsi = rgb_feats.size(0)

        if not self.args.ablate_rgb:
            if self.args.add_rgb_dropout:
                rgb_x = self.rgb_dropout(rgb_feats)
            rgb_x = self.visual_fc_rgb(rgb_feats).reshape(
                bsi, self.num_imgs, -1)

        if not self.args.ablate_depth:
            if self.args.add_depth_dropout:
                depth_x = self.depth_dropout(depth_feats)
            depth_x = self.visual_fc_depth(depth_feats).reshape(
                bsi, self.num_imgs, -1)

        if not self.args.ablate_rgb and not self.args.ablate_depth:
            cat_x = torch.cat((rgb_x, depth_x), dim=-1)
            vis_x = self.visual_merge(cat_x)
        elif not self.args.ablate_rgb:
            vis_x = rgb_x
        elif not self.args.ablate_depth:
            vis_x = depth_x

        if self.args.add_visual_dropout:
            vis_x = self.visual_dropout(vis_x)

        attention_mask = self.mask
        vis_rel_x = self.waypoint_TRM(
            vis_x, attention_mask=attention_mask
        ) 

        if self.args.add_visual_dropout:
            vis_rel_x = self.visual_dropout(vis_rel_x)

        vis_logits = self.vis_classifier(vis_rel_x) 
        vis_logits = vis_logits.reshape(
            bsi, self.num_angles, self.n_classes) 

        # heatmap offset (each image is pointing at the middle)
        vis_logits = torch.cat(
            (vis_logits[:,self.args.HEATMAP_OFFSET:,:], vis_logits[:,:self.args.HEATMAP_OFFSET,:]),
            dim=1)

        return vis_logits


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
