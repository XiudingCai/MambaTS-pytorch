# MambaTS

The repo is the official implementation for the paper: [MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting](http://arxiv.org/abs/2405.16440).

Key codes:

* For the architecture design of MambaTS, please refer primarily to `models/MambaTS.py`.
* For variable permutation training (VPT), please focus on the `random_shuffle` and `unshuffle` function in `utils/masking.py`.
* For variable-aware scan along time (VAST), please refer mainly to `layers/mamba_ssm/mixer2_seq_simple.py`.

Recently, we've also released a repo tracking the latest developments in Mamba. If you're interested, you can check it out at [Awesome-Mamba-Collection](https://github.com/XiudingCai/Awesome-Mamba-Collection) and enjoy it.

## Usage

1. Install Python 3.11. For convenience, execute the following command.

   ```
   pip install -r requirements.txt
   ```

2. For setting up the Mamba environment, please refer to https://github.com/state-spaces/mamba. Here is a simple instruction on Linux system,

   ```
   pip install causal-conv1d>=1.2.0
   pip install mamba-ssm
   ```

3. Prepare Data. You can obtain the well pre-processed datasets from public channel like [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/), Then place the downloaded data in the folder`./dataset`. 

4. Train and evaluate model. We provide the experiment scripts for MambaTS under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

   ```
   # long-term forecast for ETTm2 dataset
   bash ./scripts/long_term_forecast/MambaTS_ETTm2.sh
   ```

## Acknowledgement

This library is constructed based on the following repos:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library
- Mamba: https://github.com/state-spaces/mamba

All the experiment datasets are public, and we obtain them from the following links:

- Long-term Forecasting: https://github.com/thuml/Autoformer and https://github.com/thuml/iTransformer.

We also greatly appreciate the [python-tsp](https://github.com/fillipe-gsm/python-tsp) for providing efficient solvers for the Asymmetric Traveling Salesperson Problem (ATSP).

We extend our sincere thanks for their excellent work and repositories!

## Citation

If you find this repo useful, please consider citing our paper.

```
@article{cai2024mambats,
  title={MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting},
  author={Cai, Xiuding and Zhu, Yaoyao and Wang, Xueyao and Yao, Yu},
  journal={arXiv preprint arXiv:2405.16440},
  year={2024}
}
```