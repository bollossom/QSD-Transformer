# 执行训练:
cd mmdet3/tools
bash dist_train.sh ../configs/sdt_mask_rcnn/mask-rcnn_sdt-t_fpn_1x_coco.py 1

# 验证网络
bash dist_test.sh  ../configs/sdt_mask_rcnn/mask-rcnn_sdt-t_fpn_1x_coco.py checkpoint.pth  ../work_dirs/t1_adamw_0.440_15m.pth  1
