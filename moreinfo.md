
# Additional Installation Guide

This document provides **granular steps** to set up the TV³S environment beyond the basic installation.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Ashesham/TV3S.git
cd VSFM
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

## 4. Prepare `deps/` Folder

```bash
mkdir deps && cd deps
```

---

### 4.1 Build & Install MMCV (v1.3.0)

```bash
git clone --branch v1.3.0 https://github.com/open-mmlab/mmcv.git
cd mmcv
export MMCV_WITH_OPS=1
export FORCE_MLU=1
python setup.py develop    # or python setup.py install
```

> **Troubleshooting**
>
> * If you see build errors:
>
>   ```bash
>   pip install numpy==1.26.3
>   ```
> * Verify your GCC version is compatible (≥ 7.5).

---

## 6. Install Other Python Dependencies

```bash
pip install timm==0.4.12
pip install einops
pip install matplotlib ipython
pip install fast-pytorch-kmeans
pip install psutil
pip install yapf==0.40.1
```

> **Note:**
> You may also need to install **conv1d** or other ops for `mmaba_ssm`—please revisit once the core setup is complete.

---

## 7. Link Your Dataset

```bash
# From the project root:
ln -s /Your/Dataset/Path/ .
```

---

## 8. Pretrained Weights

Place SegFormer B1 weights in:

```
pretrained/segformer/mit_b1.pth
```

---

## 9. Temporary Files Management

To prevent accumulation of ephemeral files during underence, set up a temporary directory:

1. Create a dedicated temp directory:

   ```bash
   mkdir ~/tv3s_tmp
   export TMPDIR=~/tv3s_tmp
   ```
2. Periodically clean:

   ```bash
   rm -rf ~/tv3s_tmp/*
   ```

---

