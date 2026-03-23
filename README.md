<div align="center">
    <h2>
        Unsupervised Change Detection of Heterogeneous Remote Sensing Images via Dynamic Mask-guided Reconstruction
    </h2>
</div>
<br>

<div align="center">
  <img src="MaskUCD.jpg" width="800"/>
</div>

## Abstract
Unsupervised change detection (CD) in heterogeneous remote sensing images is intrinsically difficult due to severe sensor-specific discrepancies. In the absence of ground truth, these discrepancies result in ambiguous optimization objectives that make it difficult for models to distinguish true land-cover changes from modality-driven pseudo-changes. To address these challenges, we propose MaskUCD, a novel unsupervised framework that analyzes change based on latent features, and is driven by a progressive mask-guided refinement strategy. Unlike conventional methods relying on static constraints or passive alignment strategies, MaskUCD establishes an active, reconstruction-driven feedback loop. This mechanism dynamically generates a guidance mask to enforce feature alignment in mask-unchanged regions and divergence in mask-changed regions, which iteratively decouples real changes from modality differences. To support this iterative optimization, we design a specialized asymmetric autoencoder with a hybrid encoder architecture. By integrating multi-scale frequency analysis and global context modeling, this design creates a functional bottleneck that forces the network to filter out modality-specific noise while preserving semantic integrity. Consequently, the progressively refined mask provides increasingly cleaner supervision, yielding a converged and interpretable difference map derived from the optimized feature discrepancies, from which the final high-quality change map is obtained. Extensive experiments demonstrate that MaskUCD achieves state-of-the-art performance and superior robustness.


## Installation
### Step1: Clone the MaskUCD repository

```bash
git clone https://github.com/DGUTAiNET/MaskUCD.git
cd MaskUCD
```

### Step2:  Create and activate a new conda environment

```bash
conda create -n MaskUCD
conda activate MaskUCD
```

### Step3: Install Dependencies
Install PyTorch firstly, we recommend using the pytorch>=2.0, cuda>=11.8.
***
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
cd selective_scan && pip install .
```

## Run

```bash
python main.py
```
## Acknowledgment

This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)) and RSMamba ([paper](https://arxiv.org/abs/2403.19654),[code](https://github.com/KyanChen/RSMamba))
, thanks for their excellent works.
