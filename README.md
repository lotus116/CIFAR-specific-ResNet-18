# CIFAR-specific ResNet-18

A compact, reproducible CIFAR-10 image-classification project built entirely from basic
PyTorch layers. The final model reaches **95.88% validation accuracy** and **95.66% test
accuracy**, without pretrained weights or a model-library architecture.

## Network design

The model adapts ResNet-18 to 32×32 images:

- a 3×3, stride-1 stem preserves spatial information;
- four residual stages use 64, 128, 256, and 512 channels;
- each stage contains two basic residual blocks;
- stride-2 projection shortcuts perform downsampling;
- global average pooling replaces a large fully connected head;
- the model has 11,173,962 parameters.

`model.py` exposes a no-argument `build_model()` factory. Importing it does not train,
download data, or load a checkpoint.

## Training recipe

The final model was trained from scratch for 100 epochs with batch size 256 and seed
2026. Optimization uses SGD, learning rate 0.1, momentum 0.9, Nesterov momentum,
weight decay 5e-4, and cosine learning-rate decay. Training augmentation and
regularization comprise:

- reflection-padded random crops and horizontal flips;
- RandAugment (2 operations, magnitude 9);
- random erasing with probability 0.25;
- Mixup with alpha 0.2, applied to half of the batches;
- label smoothing of 0.1.

A cautious early-stopping rule is available: training must reach epoch 80, then stops
only after 20 epochs without a validation gain greater than 0.02 percentage points.
The final run continued to epoch 100 because validation accuracy kept improving; its
best checkpoint occurred at epoch 97.

## Results

| Configuration | Validation | Test | Gaussian-noise proxy | Occlusion proxy |
|---|---:|---:|---:|---:|
| ResNet-18 + crop/flip, 50 epochs | 94.12% | 93.36% | 44.64% | 79.90% |
| Robust recipe, 60 epochs | 95.32% | 94.79% | 67.40% | 88.50% |
| Robust recipe, 100 epochs | **95.88%** | **95.66%** | **70.28%** | **89.48%** |

The robustness proxies are deterministic transformations of the held-out validation
split: Gaussian noise with standard deviation 0.15 in normalized space and an 8×8
central occlusion. They are used only for post-training analysis, never for checkpoint
selection.

![Validation curves](training_curves.png)

See [REPORT.md](REPORT.md) or [REPORT_ZH.md](REPORT_ZH.md) for the full experiment
summary.

## Reproduce

```powershell
conda env create -f environment.yml
conda activate cnn
python train_experiments.py --variant robust --epochs 100 --min-epochs 80 --patience 20
```

CIFAR-10 is downloaded automatically. The fixed split contains 45,000 training and
5,000 validation images. Evaluation uses `ToTensor()` followed by normalization with
mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`.

## Load the released checkpoint

```python
import torch
from model import build_model

model = build_model()
state = torch.load("model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

## Repository contents

- `model.py`: network definition and model factory;
- `model.pt`: best 100-epoch checkpoint;
- `train_experiments.py`: reproducible training and robustness evaluation;
- `experiments/`: epoch histories and final metrics;
- `REPORT.md` / `REPORT_ZH.md`: English and Chinese reports;
- `environment.yml`: tested CUDA 12.8 environment.

The model and experiments are released for research and educational use. CIFAR-10 is
subject to its own dataset terms.
