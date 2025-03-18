
### Train 

Train:

```shell
torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 256 \
  --blr 6e-4 \
  --warmup_epochs 5 \
  --epochs 200 \
  --model Efficient_Spiking_Transformer_s \
  --data_path /your/data/path \
  --output_dir outputs/T1 \
  --log_dir outputs/T1 \
  --model_mode ms \
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
