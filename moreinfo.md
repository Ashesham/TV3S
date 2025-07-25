
# Additional Installation Guide

This document provides **granular steps** to set up the TV³S environment beyond the basic installation.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Ashesham/TV3S.git
cd TV3S
```
---

## 2. Create & Activate Conda Environment

```bash
conda create -n tv3s python=3.10
conda activate tv3s
```

---

## 3. Install PyTorch (CUDA 12.1)

```bash
pip install \
  torch==2.2.0 \
  torchvision==0.17.0 \
  torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Build and install MMCV with custom ops

```bash
cd 3rdparty/mmcv/
export MMCV_WITH_OPS=1
export FORCE_MLU=1
python setup.py develop    # or `python setup.py install`
cd ../..
```

> **Troubleshooting**
> 
> * Verify your GCC version is compatible (≥ 7.5).

---

## 5. Install Other Python Dependencies

```bash
pip install timm==0.4.12
pip install einops
pip install matplotlib ipython
pip install fast-pytorch-kmeans
pip install psutil
pip install yapf==0.40.1
pip install mmsegmentation==0.11.0 mmengine==0.10.7
pip install numpy==1.26.3
pip install ninja==1.11.1.3

```

---

# 6. Set up the causal-conv1d library
```bash
cd 3rdparty
git clone https://github.com/Dao-AILab/causal-conv1d.git -b v1.5.0.post8
export MAX_JOBS=16
cd causal-conv1d/
python setup.py install
cd ..
```
---

# 7. Install the mamba module
```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba
python setup.py install
cd ..
```

## 7. Link Your Dataset

```bash
# From the project root:
mkdir -p data/vspw/
ln -s /path/to/VSPW_480p data/vspw/
```

---

## 8. Pretrained Weights

Place SegFormer B1 weights in:

```
pretrained/segformer/mit_b1.pth
```

---

## 9. Temporary Files Management

To prevent accumulation of files during inference, set up a temporary directory:

1. Create a dedicated temp directory:

```bash
mkdir ~/tv3s_tmp
   ```bash
   mkdir ~/tv3s_tmp
   export TMPDIR=~/tv3s_tmp # Or add to .bashrc
   ```
2. Periodically clean:

   ```bash
   rm -rf ~/tv3s_tmp/*
   ```

---

