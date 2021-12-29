# Hire-Wave-MLP.pytorch


## Implementation of [Hire-MLP: Vision MLP via Hierarchical Rearrangement](https://arxiv.org/pdf/2108.13341.pdf) and [An Image Patch is a Wave: Phase-Aware Vision MLP](https://arxiv.org/pdf/2111.12294.pdf)


## Results and Models

### Hire-MLP on ImageNet-1K Classification

| Model                | Parameters | FLOPs    | Top 1 Acc. | Log | Ckpt |
| :------------------- | :--------: | :------: | :--------: | :------: | :------: |
| Hire-MLP-Tiny        | 18M        |  2.1G    |  79.7%     | [github](https://github.com/ggjy/Hire-Wave-MLP.pytorch/releases/download/log/hire-mlp-tiny-log.txt) | [github](https://github.com/ggjy/Hire-Wave-MLP.pytorch/releases/download/log/hire_mlp_tiny.pth) |
| Hire-MLP-Small       | 33M        |  4.2G    |  82.1%     | [github](https://github.com/ggjy/Hire-Wave-MLP.pytorch/releases/download/log/hire-mlp-small-log.txt) | [github](https://github.com/ggjy/Hire-Wave-MLP.pytorch/releases/download/log/hire_mlp_small.pth) |
| Hire-MLP-Base        | 58M        |  8.1G    |  83.2%     | [github](https://github.com/ggjy/Hire-Wave-MLP.pytorch/releases/download/log/hire-mlp-base-log.txt) | [github](https://github.com/ggjy/Hire-Wave-MLP.pytorch/releases/download/log/hire_mlp_base.pth) |
| Hire-MLP-Large       | 96M        |  13.4G   |  83.8%     | | |



## Usage

### Install

- PyTorch (1.7.0)
- torchvision (0.8.1)
- timm (0.3.2)
- torchprofile
- mmcv (v1.3.0)
- mmdetection (v2.11)
- mmsegmentation (v0.11)

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is:

```
│path/to/imagenet/
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

### Training

To train Hire-MLP-Tiny on ImageNet-1K on a single node with 8 gpus:

```python -m torch.distributed.launch --nproc_per_node=8 train.py --data-path /your_path_to/imagenet/ --output_dir /your_path_to/output/ --model hire_mlp_tiny --batch-size 256 --apex-amp --input-size 224 --drop-path 0.0 --epochs 300 --test_freq 50 --test_epoch 260 --warmup-epochs 20 --warmup-lr 1e-6  --no-model-ema```

To train Hire-MLP-Base on ImageNet-1K on a single node with 8 gpus:

```python -m torch.distributed.launch --nproc_per_node=8 train.py --data-path /your_path_to/imagenet/ --output_dir /your_path_to/output/ --model hire_mlp_base --batch-size 128 --apex-amp --input-size 224 --drop-path 0.2 --epochs 300 --test_freq 50 --test_epoch 260 --warmup-epochs 20 --warmup-lr 1e-6  --no-model-ema```

### Evaluation

To evaluate a pre-trained Hire-MLP-Tiny on ImageNet validation set with a single GPU:

```python -m torch.distributed.launch --nproc_per_node=1 train.py --data-path /your_path_to/imagenet/ --output_dir /your_path_to/output/ --batch-size 256 --input-size 224 --model hire_mlp_tiny --apex-amp --no-model-ema --resume /your_path_to/hire_mlp_tiny.pth --eval```



## Acknowledgement
This repo is based on [DeiT](https://github.com/facebookresearch/deit), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [CycleMLP](https://github.com/ShoufaChen/CycleMLP) and [AS-MLP](https://github.com/svip-lab/AS-MLP).


## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{guo2021hire,
  title={Hire-mlp: Vision mlp via hierarchical rearrangement},
  author={Guo, Jianyuan and Tang, Yehui and Han, Kai and Chen, Xinghao and Wu, Han and Xu, Chao and Xu, Chang and Wang, Yunhe},
  journal={arXiv preprint arXiv:2108.13341},
  year={2021}
}
```

```bibtex
@article{tang2021image,
  title={An Image Patch is a Wave: Phase-Aware Vision MLP},
  author={Tang, Yehui and Han, Kai and Guo, Jianyuan and Xu, Chang and Li, Yanxi and Xu, Chao and Wang, Yunhe},
  journal={arXiv preprint arXiv:2111.12294},
  year={2021}
}
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
