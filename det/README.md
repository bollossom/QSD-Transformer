Download the pretrained model

Pre-trained ckpts 15M_w4: [here](https://huggingface.co/Xuerui123/QSD_Transformer).

Pre-trained ckpts 55M_w4: [here](https://huggingface.co/Xuerui123/QSD_Transformer).

# Train:
cd mmdet3/tools
bash dist_train.sh ../configs/q_sformer_mask_rcnn/mask-rcnn_qformer_fpn_1x_coco.py 1

# Test
bash dist_test.sh  ../configs/q_sformer_mask_rcnn/mask-rcnn_qformer_fpn_1x_coco.py checkpoint ../work_dirs/t1_adamw_0.440_15m.pth  1
