# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=1.0,model_mode='Q_trick'):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    if model_mode=='new_design':
        num_layers = len(model.block3)  + 1
    else:
        num_layers = len(model.block3) + len(model.block4) + 1
    

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad: # 仅针对需要利用梯度进行更新的参数
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
    
        if group_name not in param_group_names:
            N=9
            scale_exp = N + 1 - layer_id
            this_scale = layer_scales[layer_id] ** scale_exp
            group_name = f'layer{layer_id}_' + group_name
            
            dbg = f'[layer {layer_id}][sc = {this_scale} ** {scale_exp}]'
            print('dbg',dbg)
            print("++++++++++")
            
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    exit(0)
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('block'):
        #return int(name.split('.')[1]) + 1
        return num_layers
    else:
        return num_layers
# def get_param_groups(model, nowd_keys=(), lr_scale=0.0):
#     using_lr_scale = True
#     print(f'[get_ft_param_groups][lr decay] using_lr_scale={using_lr_scale}, ft_lr_scale={lr_scale}')
#     para_groups, para_groups_dbg = {}, {}
#     for name, para in model.named_parameters():
#         if not para.requires_grad:
#             continue  # frozen weights
#         if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
#             wd_scale, group_name = 0., 'no_decay'
#         else:
#             wd_scale, group_name = 1., 'decay'
        
#         if using_lr_scale:
#             layer_id, scale_exp = model.get_layer_id_and_scale_exp(name)
#             group_name = f'layer{layer_id}_' + group_name
#             this_lr_scale = lr_scale ** scale_exp
#             dbg = f'[layer {layer_id}][sc = {lr_scale} ** {scale_exp}]'
#         else:
#             this_lr_scale = 1
#             dbg = f'[no scale]'
        
#         if group_name not in para_groups:
#             para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': this_lr_scale}
#             para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': dbg}
#         para_groups[group_name]['params'].append(para)
#         para_groups_dbg[group_name]['params'].append(name)
    
#     for g in para_groups_dbg.values():
#         g['params'] = pformat(', '.join(g['params']), width=200)
#     print(this_lr_sclae)
#     exit(0)
#     print("++++++++++")
#     print(f'[get_ft_param_groups] param groups = \n{pformat(para_groups_dbg, indent=2, width=250)}\n')
#     return list(para_groups.values())