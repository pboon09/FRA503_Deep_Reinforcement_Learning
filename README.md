# FRA503 Deep Reinforcement Learning for Robotics

## Part 1: Miniconda Installation

Download Miniconda with Python 3.10 [[list of Miniconda](https://repo.anaconda.com/miniconda)]. This is just the base Python version. You can create environments with any Python version on top of it.

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh
```

Install Miniconda

```bash
bash ~/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh
```

Close and re-open your terminal window, or use:

```bash
source ~/.bashrc
```

### Verifying Miniconda Installation

```bash
conda list
```

If a list of installed packages appears, then the installation was successful! 🎉

---

## Part 2: Isaac Sim Installation

**Check your CUDA Version**

```bash
nvidia-smi
```

### Option A: Isaac Sim 4.5.0 using Isaac Lab 2.1.0

Please refer to the official installation guide [[link](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)]

**Create Conda Environment**

```bash
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab
```

**Install PyTorch if you're using CUDA 11**

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

**Install PyTorch if you're using CUDA 12**

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

**Install Isaac Sim 4.5.0**

```bash
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

---

### Option B: Isaac Sim 5.1.0 using Isaac Lab Latest Version

Please refer to the official installation guide [[link](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)]

**Create Conda Environment**

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
```

**Install PyTorch if you're using CUDA 12**

```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

**Install Isaac Sim 5.1.0**

```bash
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

---

### Verifying Isaac Sim Installation

```bash
isaacsim
```

- Takes a long time on first run, so be patient!
- Type `Yes` to accept EULA

If the simulator window opens, then the installation was successful! 🎉

---

## Part 3: Isaac Lab Installation

### Option A: Isaac Lab 2.1.0

```bash
cd ~
git clone -b release/2.1.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### Option B: Isaac Lab Latest Version

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### Install Dependencies

```bash
sudo apt install cmake build-essential
```

### Install Isaac Lab

```bash
./isaaclab.sh --install
```

### Verifying Isaac Lab Installation

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

If you see a black viewport window, then the installation was successful! 🎉

---

## Part 4: Environment Activation

### Activate Environment

```bash
conda activate env_isaaclab
```

### Deactivate Environment

```bash
conda deactivate
```

### Check GPU & PyTorch

```bash
# GPU status
nvidia-smi

# PyTorch verification
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.is_available()}')"
```

---

## Part 5: Test Training

### Activate Environment

```bash
cd ~/IsaacLab
conda activate env_isaaclab
```

### Train Robot 
Remove `--headless` to open GUI (headless for faster training)

```bash
# Ant
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless

# Robot dog
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless

# Humanoid
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Humanoid-v0 --headless

# Cartpole
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless
```

### Play Trained Policy
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Ant-v0

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-Anymal-C-v0

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Humanoid-v0

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Cartpole-v0
```

### View Training Results

```bash
# Check trained model location
ls logs/rsl_rl/ant/
ls logs/rsl_rl/anymal_c_rough/
ls logs/rsl_rl/humanoid/
ls logs/rsl_rl/cartpole/

# View the latest training folder
ls -la logs/rsl_rl/ant/$(ls -t logs/rsl_rl/ant/ | head -1)/
```

### Delete Training Results

```bash
# Delete specific training run
rm -rf logs/rsl_rl/ant/
rm -rf logs/rsl_rl/anymal_c_rough/
rm -rf logs/rsl_rl/humanoid/
rm -rf logs/rsl_rl/cartpole/

# Delete all training logs
rm -rf logs/
```

### List Available Tasks

```bash
python scripts/environments/list_envs.py
```

---

## Troubleshooting

### Remove and Recreate Environment

```bash
conda deactivate
rm -rf ~/miniconda3/envs/env_isaaclab
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab
```

### Check System Specs

```bash
# CPU info
lscpu | grep "Model name"

# RAM
free -h

# GPU and VRAM
nvidia-smi

# Summary view
sudo apt install neofetch
neofetch
```

---

## A* Star ARTC Machine Spec

| Component | Spec |
|-----------|------|
| **OS** | Ubuntu 24.04.3 LTS |
| **CPU** | Intel Core Ultra 9 285K (24 cores) @ 4.1GHz |
| **RAM** | 128 GB |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell |
| **VRAM** | 96 GB |
| **CUDA** | 13.0 |
| **Architecture** | sm_120 (Blackwell) |