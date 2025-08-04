import torch
import torch.nn as nn
import numpy as np
import utils

from transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def WP_predict(mode, args, predictor, rgb_feats, depth_feats, rgb_down_feats=None, depth_down_feats=None):
    ''' predicting the waypoint probabilities '''
    vis_logits = predictor(rgb_feats, depth_feats, rgb_down_feats, depth_down_feats)
    # entry-wise probabilities
    if args.train_use_sigmoid:
        vis_logits = torch.sigmoid(vis_logits)
        vis_probs = vis_logits
    else:
        # Original
        vis_probs = torch.sigmoid(vis_logits) # TODO：为什么训练的时候不加sigmoid?

    if mode == 'train':
        return vis_logits
    elif mode == 'eval':
        return vis_probs, vis_logits


class WaypointPredictor(nn.Module):
    def __init__(self, args=None, rgb_input_dim=1024, depth_input_dim=1024, hidden_dim=768, n_classes=12):
        super(WaypointPredictor, self).__init__()
        self.args = args
        self.batchsize = args.BATCH_SIZE
        self.num_angles = args.ANGLES
        self.num_imgs = args.NUM_IMGS
        self.n_classes = n_classes

        if self.args.wp_add_before_envdrop:
            self.before_envdrop = nn.Dropout(self.args.wp_before_envdrop_prob)
        if self.args.wp_add_after_envdrop:
            self.after_envdrop = nn.Dropout(self.args.wp_after_envdrop_prob)
        
        self.visual_fc_rgb = nn.Sequential(
            nn.Linear(rgb_input_dim, hidden_dim),
            nn.ReLU(True),
            BertLayerNorm(hidden_dim)
        )

        if self.args.depth_encoder_type == 'dinov2':
            self.visual_fc_depth = nn.Sequential(
                nn.Linear(depth_input_dim, hidden_dim),
                nn.ReLU(True),
                BertLayerNorm(hidden_dim)
            )
        elif self.args.depth_encoder_type == 'resnet':
            self.visual_fc_depth = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod([128,4,4]), hidden_dim),
                nn.ReLU(True),
            )

        if self.args.include_down_views:
            self.visual_fc_rgb_down = nn.Sequential(
                nn.Linear(rgb_input_dim, hidden_dim),
                nn.ReLU(True),
                BertLayerNorm(hidden_dim)
            )
            if self.args.depth_encoder_type == 'dinov2':
                self.visual_fc_depth_down = nn.Sequential(
                    nn.Linear(depth_input_dim, hidden_dim),
                    nn.ReLU(True),
                    BertLayerNorm(hidden_dim)
                )
            elif self.args.depth_encoder_type == 'resnet':
                self.visual_fc_depth_down = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod([128,4,4]), hidden_dim),
                    nn.ReLU(True),
                )

            self.rgb_ln = BertLayerNorm(hidden_dim)
            self.depth_ln = BertLayerNorm(hidden_dim)
        
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
        # self.mergefeats_LayerNorm = BertLayerNorm(
        #     hidden_dim,
        #     eps=layer_norm_eps
        # )

        self.mask = utils.get_attention_mask(
            num_imgs=self.num_imgs,
            neighbor=args.TRM_NEIGHBOR).to(device)

        self.vis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                int(n_classes*(self.num_angles/self.num_imgs))),
        )

        # 添加可学习的position embedding
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_imgs, hidden_dim))
        nn.init.normal_(self.position_embeddings, mean=0, std=0.02)

    def forward(self, rgb_feats, depth_feats, rgb_down_feats=None, depth_down_feats=None):
        # bsi = rgb_feats.size(0) // self.num_imgs
        bsi = rgb_feats.size(0)
        
        if self.args.wp_add_before_envdrop:
            rgb_feats = self.before_envdrop(rgb_feats)
            depth_feats = self.before_envdrop(depth_feats)
            if rgb_down_feats is not None:
                rgb_down_feats = self.before_envdrop(rgb_down_feats)
            if depth_down_feats is not None:
                depth_down_feats = self.before_envdrop(depth_down_feats)

        # rgb_x = self.visual_1by1conv_rgb(rgb_feats)
        rgb_x = self.visual_fc_rgb(rgb_feats)
        
        if rgb_down_feats is not None:
            rgb_down_x = self.visual_fc_rgb(rgb_down_feats)
            rgb_x = rgb_x + rgb_down_x
            rgb_x = self.rgb_ln(rgb_x)

        # depth_x = self.visual_1by1conv_depth(depth_feats)
        if len(depth_feats.shape) == 5:
            # load from pre-extracted features (resnet), (bs, 12, 128, 4, 4)
            depth_feats = depth_feats.view(bsi*self.num_imgs, *depth_feats.shape[2:])
        depth_x = self.visual_fc_depth(depth_feats)
        if self.args.depth_encoder_type == 'resnet':
            bsi_depth = depth_x.size(0) // self.num_imgs
            depth_x = depth_x.reshape(bsi_depth, self.num_imgs, -1)
    
        if depth_down_feats is not None:
            depth_down_x = self.visual_fc_depth(depth_down_feats)
            if self.args.depth_encoder_type == 'resnet':
                bsi_depth_down = depth_down_x.size(0) // self.num_imgs 
                depth_down_x = depth_down_x.reshape(bsi_depth_down, self.num_imgs, -1)
            depth_x = depth_x + depth_down_x
            depth_x = self.depth_ln(depth_x)

        if self.args.wp_add_after_envdrop:
            rgb_x = self.after_envdrop(rgb_x)
            depth_x = self.after_envdrop(depth_x)

        vis_x = self.visual_merge(
            torch.cat((rgb_x, depth_x), dim=-1)
        )
        
        # 添加position encoding
        # vis_x = vis_x + self.position_embeddings

        # attention_mask = self.mask.repeat(bsi,1,1,1)
        attention_mask = self.mask
        
        vis_rel_x = self.waypoint_TRM(
            vis_x, attention_mask=attention_mask
        ) # vis_rel_x: [8,12,768]

        vis_logits = self.vis_classifier(vis_rel_x) # vis_logits: [8,12,120]
        vis_logits = vis_logits.permute(0,2,1) # [8,120,12]

        # heatmap offset (each image is pointing at the middle)
        vis_logits = torch.cat(
            (vis_logits[:,self.args.HEATMAP_OFFSET:,:], vis_logits[:,:self.args.HEATMAP_OFFSET,:]),
            dim=1)

        return vis_logits

    def forward_without_depth(self, rgb_feats, rgb_down_feats=None):
        bsi = rgb_feats.size(0) // self.num_imgs
        
        if self.args.wp_add_before_envdrop:
            rgb_feats = self.before_envdrop(rgb_feats)
            if rgb_down_feats is not None:
                rgb_down_feats = self.before_envdrop(rgb_down_feats)
        
        rgb_x = self.visual_fc_rgb(rgb_feats).reshape(
            bsi, self.num_imgs, -1)
        
        if rgb_down_feats is not None:
            rgb_down_x = self.visual_fc_rgb(rgb_down_feats).reshape(
                bsi, self.num_imgs, -1)
            rgb_x = rgb_x + rgb_down_x

        if self.args.wp_add_after_envdrop:
            rgb_x = self.after_envdrop(rgb_x)

        vis_x = rgb_x
        
        attention_mask = self.mask.repeat(bsi,1,1,1)
        vis_rel_x = self.waypoint_TRM(
            vis_x, attention_mask=attention_mask
        ) # vis_rel_x: [8,12,768]

        vis_logits = self.vis_classifier(vis_rel_x) # vis_logits: [8,120,12]
        
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
