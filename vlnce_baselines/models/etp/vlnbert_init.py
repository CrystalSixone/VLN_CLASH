import torch


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'bert_config/xlm-roberta-base'
    else:
        cfg_name = 'bert_config/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(config=None):
    
    from transformers import PretrainedConfig
    if config.model_name == "etp-nav":
        from vlnce_baselines.models.etp.vilmodel_cmt import GlocalTextPathNavCMT
    elif config.model_name == "scalevln":
        from vlnce_baselines.models.etp.vilmodel_scalevln import GlocalTextPathNavCMT
    else:
        raise ValueError(f"Invalid model name: {config.model_name}")

    print(f"Using model: {config.model_name}")

    model_class = GlocalTextPathNavCMT

    model_name_or_path = config.pretrained_path
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        if "scaleVLN_ft" in model_name_or_path:
            ckpt_weights = ckpt_weights['vln_bert']['state_dict']
            for k, v in ckpt_weights.items():
                if k.startswith('vln_bert.'):
                    new_ckpt_weights['bert.' + k[9:]] = v
                else:
                    new_ckpt_weights[k] = v
        else:
            for k, v in ckpt_weights.items():
                if k.startswith('module'):
                    new_ckpt_weights[k[7:]] = v
                if 'sap_head' in k:
                    new_ckpt_weights['bert.' + k] = v
                else:
                    new_ckpt_weights[k] = v
    
    if config.task_type in ['r2r']:
        cfg_name = 'bert_config/bert-base-uncased'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    vis_config.max_action_steps = 100
    # vis_config.image_feat_size = 512
    vis_config.use_depth_embedding = config.use_depth_embedding
    vis_config.depth_feat_size = 128
    vis_config.angle_feat_size = 4

    vis_config.num_l_layers = config.num_l_layers if hasattr(config, 'num_l_layers') else 9
    vis_config.num_pano_layers = config.num_pano_layers if hasattr(config, 'num_pano_layers') else 2
    vis_config.num_x_layers = config.num_x_layers if hasattr(config, 'num_x_layers') else 4
    vis_config.graph_sprels = config.use_sprels
    vis_config.glocal_fuse = 'global'

    vis_config.fix_lang_embedding = config.fix_lang_embedding
    vis_config.fix_pano_embedding = config.fix_pano_embedding

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False

    # ScaleVLN
    vis_config.image_feat_size = config.CLIP_ENCODER.output_size # 1024 for scaleVLN (ViT-H/14)
    vis_config.rgb_clip_model = config.CLIP_ENCODER.model
    
    # global and local branch
    vis_config.add_local_branch = config.add_local_branch
    
    # GOAT  
    vis_config.do_back_txt = config.causal.do_back_txt
    vis_config.do_back_img = config.causal.do_back_img
    vis_config.do_front_txt = config.causal.do_front_txt
    vis_config.do_front_local = config.causal.do_front_local
    vis_config.do_front_global = config.causal.do_front_global
    vis_config.do_add_method = 'door'
    vis_config.front_lg_door = True
    vis_config.front_self_attn = True
    vis_config.do_back_img_type = config.causal.do_back_img_type
    
    # Dropout
    vis_config.add_next_action_pred_dropout = config.add_next_action_pred_dropout
    vis_config.env_dropout_prob = config.env_dropout_prob if hasattr(config, 'env_dropout_prob') else 0.4
    vis_config.add_depth_dropout = config.add_depth_dropout if hasattr(config, 'add_depth_dropout') else False
    vis_config.add_global_dropout = config.add_global_dropout if hasattr(config, 'add_global_dropout') else False
    
    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
        
    return visual_model
