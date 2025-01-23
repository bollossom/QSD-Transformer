python -m torch.distributed.launch  --nproc_per_node=6 \
  --master_port=25678 \
  tools/train.py \
  configs/q_sformer_seg/fpn_sdtv3_512x512_15M_ade20k.py 
  