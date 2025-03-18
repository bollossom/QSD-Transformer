
### Train 

Train:

```shell
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
```



### Data Prepare

ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
