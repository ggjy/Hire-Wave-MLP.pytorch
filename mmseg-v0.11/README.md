Based on [MMSegmentation-V0.11.0](https://github.com/open-mmlab/mmsegmentation/releases/tag/v0.11.0), Documentation of MMSeg: https://mmsegmentation.readthedocs.io/

## Installation

```
pip install addict;
pip install yapf;
pip install cython;
pip install opencv-python;
pip install pytest-runner;
pip install terminaltables==3.1.0;
pip install mmpycocotools;
cd mmcv-1.3.0;
MMCV_WITH_OPS=1 pip install -e . --disable-pip-version-check;
cd ../mmseg-v0.11; pip install -e . --disable-pip-version-check;
```

Or you can refer to [get_started.md](docs/get_started.md#installation) for installation and dataset preparation.

## Get Started

### Training

To train Semantic-FPN based Hire-MLP-Small on ADE20K on a single node with 8 gpus:

```
cd mmseg-v0.11/; python -m torch.distributed.launch --nproc_per_node=8 tools/train.py configs/sem_fpn/hire_mlp_small_512x512_ade20k.py --gpus 8 --launcher pytorch --work-dir /your_path_to/<SEG_CHECKPOINT_FILE>/
```

### Evaluation

```
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU
```

Or you can see [train.md](docs/train.md) and [inference.md](docs/inference.md) for the basic usage of MMSegmentation.
