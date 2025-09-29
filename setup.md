### Create environment (or use `conda` instead of `micromamba` by swapping them in the code)
```bash
micromamba create --prefix spine python=3.11.13
```

### Install [VGGT](https://github.com/facebookresearch/vggt/blob/main/docs/package.md)
```bash
git clone git@github.com:facebookresearch/vggt.git`
cd vggt
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
pip install -e 
```

### Install [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html)

Follow these steps:
1. Install `cuda-toolkit`
- For `conda`
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```
- For `micromamba`
```bash
micromamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit`
```
2. `TinyCUDA`
```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

3. Git clone and install Nerfstudio
```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install -e .
ns-install-cli
```


### Optional: Install [hloc](https://github.com/cvg/Hierarchical-Localization#:~:text=git%20clone%20%2D%2Drecursive%20https%3A//github.com/cvg/Hierarchical%2DLocalization/%0Acd%20Hierarchical%2DLocalization/%0Apython%20%2Dm%20pip%20install%20%2De%20.)
```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
```

### Install Colmap
- `conda`
```bash
conda install conda-forge::colmap
```
- `micromamba`
```bash
micromamba install conda-forge::colmap
```

#### Install Spine (after activating your virtual environment and navigating to `spine_gsplat` or `spine_nerf`)
```bash
pip install -e .
```
