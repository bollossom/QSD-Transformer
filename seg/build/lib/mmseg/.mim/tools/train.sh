# bash dist_train.sh ../configs/sdeformer_mask_rcnn/mask-rcnn_sdeformer_fpn_1x_coco.py 8
# bash dist_train.sh ../configs/sdeformer_mask_rcnn/mask-rcnn_sdeformer-s_fpn_1x_coco.py 1
# 0.48
# CUDA_VISIBLE_DEVICES=7 nohup > mask-rcnn_sdeformer_fpn_1x_coco.log ./dist_train.sh ../configs/sdeformer_mask_rcnn/mask-rcnn_sdeformer_fpn_1x_coco.py 1
# CUDA_VISIBLE_DEVICES=0,2,6 nohup > mask-rcnn_qformer-t_fpn_1x_coco_float_3.log ./dist_train.sh ../configs/q_sformer_mask_rcnn/mask-rcnn_qformer_fpn_1x_coco.py 3
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 nohup > mask-rcnn_qformer-b_fpn_1x_coco_55M.log ./dist_train.sh ../configs/q_sformer_mask_rcnn/mask-rcnn_qformer-b_fpn_1x_coco.py 6
# bash dist_train.sh ../configs/q_sformer_mask_rcnn/mask-rcnn_qformer-b_fpn_1x_coco.py 6
# bash ./dist_train.sh ../configs/EFSDTv2/fpn_sdtv3_512x512_10M_ade20k.py 4
bash ./dist_train_2.sh ../configs/q_sformer_seg/fpn_sdtv3_512x512_15M_ade20k.py 4