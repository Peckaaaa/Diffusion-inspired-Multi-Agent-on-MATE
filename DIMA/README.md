# DIMA - Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective

Code for the NIPS'25 paper *Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective*.

## Installation
### 1. Create conda environment
```bash
conda env create -f environment.yml
conda activate dima
```

### 2. (Not necessary) Install specific version of Torch and TorchVision that you prefer

```
# for example
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# torch 2.0 version and above is ok
```

### 3. Install SMAC and SMACv2 (recommend local installation)

First, you should install the engine following the instruction on this [link](https://github.com/oxwhirl/smac?tab=readme-ov-file#installing-starcraft-ii).

```
git clone https://github.com/oxwhirl/smac.git
cd smac/
pip install -e .

git clone https://github.com/oxwhirl/smacv2.git
cd smacv2
pip install -e .
```

### 4. Install MPE

```
pip install pettingzoo==1.22.2
pip install supersuit==3.7.0
```

### 5. Install MAMujoco

First, install mujoco210 and put it in the ~/.mujoco. See how to set up mujoco on this [link](https://pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html).


```
pip install gym[mujoco]
pip install patchelf

pip install 'mujoco-py<2.2,>=2.1'
pip install "Jinja2==3.0.3"
pip install "glfw==2.5.1"
pip install "Cython==0.29.28"
```

## Running a single experiment
An example for running DIMA on MAMuJoCo

```
python train.py \
    --n_workers 1 \
    --env mamujoco \
    --env_name $map_name \
    --policy_class $used_policy_class \
    --seed $seed \
    --agent_conf $agent_conf \
    --steps $steps \
    --mode $wandb_log_mode \
    --sample_temp 20
```

- ```map_name```: which scenario or map to evaluate DIMA on. For example, in MAMuJoCo, we can set map_name as *HalfCheetah-v2* to evaluate.
- ```agent_conf```: which agent splitting configure to use in MAMuJoCo, such as 2x3.
- ```steps```: the maximum environment steps in the low data regime.
- ```wandb_log_mode```: whether to enable wandb logging. Options: disabled, offline, online
- ```used_policy_class```: which continuous stochastic policy class to use. We use Gaussian by default.
- ```seed```: random seed for running this experiment.