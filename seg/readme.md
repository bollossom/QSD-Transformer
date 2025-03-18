
This folder contains the implementation of the E-Spikformer transfer learning for semantic segmentation on ADE-20K. It is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

- Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

- Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Data preparation
Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k) to prepare the ADE20k dataset.

### Training
Download the pretrained model

Pre-trained ckpts 10M: [here](https://drive.google.com/file/d/1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq/view?usp=sharing).

Pre-trained ckpts 19M: [here](https://drive.google.com/file/d/1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H/view?usp=drive_link).


Train 19M on 1 GPUs:

- `cd tools`
- `CUDA_VISIBLE_DEVICES=0,1 ./dist_train.sh ../configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k.py 1`
