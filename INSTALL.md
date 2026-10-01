# Install

Requires Python 3.10–3.11 (torch 2.1 wheels). GPU runs were done on an RTX 3050
laptop (4 GB VRAM); CPU suffices for everything except landmark training and
embedding extraction.

```powershell
# CUDA 12.1 build (as developed)
pip install -r requirements.txt

# CPU-only build
pip install -r requirements-ci.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.0+cpu torchvision==0.16.0+cpu
```

`requirements.txt` pins the exact CUDA environment (torch/torchvision `+cu121`).
`requirements-ci.txt` is the lean CPU/test set used by CI. Pretrained weights
(ResNet-50, Keypoint R-CNN) download automatically from PyTorch Hub on first use.
Datasets are NOT automated: CattleFace-RGBT via Hugging Face, sheep mirror via HF,
ReCowGnition/equine via author request — see `data/README.md`.
