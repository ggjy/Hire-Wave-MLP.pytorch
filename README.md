# Hire-Wave-MLP.pytorch


## Implementation of [Hire-MLP: Vision MLP via Hierarchical Rearrangement](https://arxiv.org/pdf/2108.13341.pdf) and [An Image Patch is a Wave: Phase-Aware Vision MLP](https://arxiv.org/pdf/2111.12294.pdf)

This repo will be complemented in one week.


## Install

- PyTorch (1.7.0)
- torchvision (0.8.1)
- timm (0.3.2)
- torchprofile
- mmcv (v1.3.0)
- mmdetection (v2.11)
- mmsegmentation (v0.11)

## Data preparation

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

## Evaluation


## Training


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
