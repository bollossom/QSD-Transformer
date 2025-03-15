# bash dist_train.sh ../configs/sdeformer_mask_rcnn/mask-rcnn_sdeformer_fpn_1x_coco.py 8
# bash dist_train.sh ../configs/sdeformer_mask_rcnn/mask-rcnn_sdeformer-s_fpn_1x_coco.py 1
# 0.48
# CUDA_VISIBLE_DEVICES=7 nohup > mask-rcnn_sdeformer_fpn_1x_coco.log ./dist_train.sh ../configs/sdeformer_mask_rcnn/mask-rcnn_sdeformer_fpn_1x_coco.py 1
CUDA_VISIBLE_DEVICES=0,2,6 nohup > mask-rcnn_qformer-t_fpn_1x_coco_float_3.log ./dist_train.sh ../configs/q_sformer_mask_rcnn/mask-rcnn_qformer_fpn_1x_coco.py 3