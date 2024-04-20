# SENeLF

# Getting The Source Code
```
# clone repo and pull submodules
git clone https://github.com/ATPAustinPeng/SENeLF
cd SENeLF
git submodule update --init --recursive
```

# Steps To Replicating My Results
1. downloading data
2. creating environments (2 total)
3. train teacher model with instant-ngp
4. data distillation with trained teacher model
5. train MobileR2L

## Downloading Data
- example data (only synthetic360 lego & rff fern)
    - `sh script/download_example_data.sh`
- all data
    - Google Drive provided by NeRF authors https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    - `scp` files into MobileR2L/dataset and unzip with `unzip [ZIP_FILE_NAME]`

## Creating Environments
### MobileR2L
```bash
# enter MobileR2L directory
cd MobileR2L

# create conda env
conda create -n r2l python==3.9
conda activate r2l
conda install pip

pip install torch torchvision torchaudio
pip install -r requirements.txt 

conda deactivate
```

### Instant-NGP
- Note, I found this environment very difficult to setup. I only have it working on:
    - A100 GPU, CUDA 11.7, SM80 (compute capability)
- Grab a GPU before performing environment setup
    - `salloc --gres=gpu:A100:1 --mem=32GB`

```bash
cd model/teacher/ngp_pl

# create conda env
conda create -n ngp_pl python==3.9
conda activate ngp_pl
conda install pip

# install torch with cuda 11.7 (pace A100 driver is this)
# for other versions check here https://pytorch.org/get-started/previous-versions/
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# specify compute capability for tiny-cuda-nn install
export TCNN_CUDA_ARCHITECTURES=80

# install tiny-cuda-nn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install torch scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+${cu117}.html

# ---install apex---
git clone https://github.com/NVIDIA/apex
cd apex

# solves https://github.com/NVIDIA/apex/issues/1735
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82

# denpendcy for apex
pip install packaging

# speeds up apex install
pip install ninja

## if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
# solves https://github.com/NVIDIA/apex/issues/1193
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
## otherwise - DID NOT WORK FOR ME, just update pip and use above!
#pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ../
# ---end installing apex---

# install other requirements
pip install -r requirements.txt

# build
pip install models/csrc/

# go to root
cd ../../../
```

## Train Teacher Model
```bash
cd model/teacher/ngp_pl

export ROOT_DIR=../../../dataset/nerf_synthetic/
python3 train.py \
     --root_dir $ROOT_DIR/lego \
     --exp_name lego \
     --num_epochs 30 --batch_size 16384 --lr 2e-2 --eval_lpips \
     --num_gpu 1
```

## Data Distillation With Trained Teacher Model
```bash
export ROOT_DIR=../../../dataset/nerf_synthetic/
python3 train.py \
    --root_dir $ROOT_DIR/lego \
    --exp_name Lego_Pseudo  \
    --save_pseudo_data \
    --n_pseudo_data 5000 \
    --weight_path ckpts/nerf/lego/epoch=29_slim.ckpt \
    --save_pseudo_path Pseudo/lego --num_gpu 1
```

## Train MobileR2L
- `salloc --gres=gpu:H100:4 --mem=64GB --ntasks-per-node=12 --time=0-04:00:00`
	- the optimal configuration I could request reasonably from PACE

```bash
# go to the MobileR2L directory
cd ../../../

# deactivate environment - ngp_pl
conda deactivate

conda activate r2l

# use 4 gpus for training: NeRF
sh pace-script/benchmarking_nerf.sh 4 lego

# train from checkpoint
sh pace-script/benchmarking_nerf_from_ckpt.sh 4 lego [FULL_CKPT_PATH]

# # render from checkpoint
# sh pace-script/benchmarking_pruned_nerf_from_ckpt.sh 4 lego [FULL_UNPRUNED_CKPT_PATH]

# use 4 gpus for training: LLFF
#sh script/benchmarking_llff.sh 4 orchids
```