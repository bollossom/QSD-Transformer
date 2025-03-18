CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 196 \
  --blr 6e-4 \
  --warmup_epochs 15 \
  --epochs 300 \
  --model spikformer_8_15M_CAFormer \
  --data_path /dataset/ImageNet2012/ \
  --output_dir ./output_dir \
  --log_dir ./log_dir \
  --dist_eval
