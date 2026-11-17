# WACV 2026 : TED-4DGS: Temporally Activated and Embedding-based Deformation for 4DGS Compression

Accepted to WACV 2026

[Cheng-Yuan Ho](https://www.linkedin.com/in/kelvincyho?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app), 
[He-Bi Yang](https://www.linkedin.com/in/hebiyang?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app ), 
[Jui Chiu Chiang](https://chiang.ccu.edu.tw/p/404-1255-68777.php?Lang=zh-tw), 
[Yu-Lun Liu](https://yulunalexliu.github.io/), 
[Wen-Hsiao Peng](https://sites.google.com/g2.nctu.edu.tw/wpeng)

[[`Arxiv`](https://www.arxiv.org/abs/2512.05446)] [[`Project Page`](https://nycu-mapl.github.io/TED-4DGS_page/)] 

## News

- This codebase additionally supports the illumination compensation setting of
  **IC-4DGS: Illumination-Compensated 4D Gaussian Splatting Under Photometric Variations** (ICIP 2026)
  [[`Paper`](https://ieeexplore.ieee.org/document/11630311)].
  See [Illumination compensation (IC-4DGS)](#illumination-compensation-ic-4dgs) for usage.


## Abstract
<p align="left">
<img src="assets/teaser.png" width=80% height=80% 
class="center">
</p>

We present TED-4DGS, a novel framework that unifies Temporally Activated mechanisms with Embedding-based Deformation for high-efficiency 4DGS compression. 
Our approach introduces learnable temporal activation parameters for each anchor to explicitly model appearance and disappearance, 
effectively handling occlusion and disocclusion. Furthermore, we incorporate an INR-based hyperprior and a channel-wise autoregressive model to achieve compact attribute compression. 
Experimental results demonstrate that TED-4DGS achieves state-of-the-art rate-distortion performance on real-world datasets, 
significantly reducing file sizes while maintaining superior rendering quality compared to existing methods.

## Performance
<p align="left">
<img src="assets/main_performance.png" width=80% height=80% 
class="center">
</p>

## Installation

Tested on Linux with Python 3.7, PyTorch 1.12.1 and CUDA 11.6.
Building the CUDA extensions requires `nvcc` (CUDA 11.6) on your `PATH`.

```bash
git clone https://github.com/NYCU-MAPL/TED-4DGS.git
cd TED-4DGS
conda env create -f environment.yml   # run from the repo root; also builds the three CUDA submodules
conda activate ted4dgs
```

`environment.yml` installs the CUDA extensions in `submodules/`
(`diff-gaussian-rasterization`, `simple-knn`, `gridencoder`).
To rebuild them manually:

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/gridencoder
```

## Data

We evaluate on [N3DV (DyNeRF)](https://github.com/facebookresearch/Neural_3D_Video) and
[HyperNeRF (vrig)](https://github.com/google/hypernerf/releases/tag/v0.1).
Place the datasets under `./data` (or edit `DATA_ROOT` at the top of each `run_<scene>.py`):

```
data/
├── N3DV/
│   └── <scene>/                  # cook_spinach, cut_roasted_beef, flame_salmon_1, flame_steak, sear_steak
│       ├── cam00.mp4 ... camXX.mp4
│       ├── poses_bounds.npy
│       └── points3D_downsample2.ply
└── hypernerf/
    └── <scene>/                  # 3dprinter, banana, broom2, chicken
        ├── camera/
        ├── rgb/
        ├── dataset.json
        ├── metadata.json
        └── points3D_downsample2.ply
```

The loader is selected automatically: a folder with `poses_bounds.npy` is read as N3DV,
and a folder with `dataset.json` is read as HyperNeRF (each HyperNeRF scene also has its own
config in `arguments/<scene>.py`, passed by the run script).
On the first N3DV run, frames are extracted from the videos into `camXX/` folders.

### Initial point cloud

Training is initialized from `points3D_downsample2.ply` in each scene folder (about 40k points per scene);
there is no random initialization fallback.
We generate it following the data preparation of [4DGaussians](https://github.com/hustvl/4DGaussians):
run COLMAP on the scene to obtain a dense point cloud, then downsample it into `points3D_downsample2.ply`.

## Training and Compression

Each scene has its own script that runs the full two-stage pipeline:

1. **Stage 1** trains the base model for 60k iterations (skipped if `chkpnt60000.pth` already exists).
2. **Stage 2** loads the base checkpoint and runs a rate sweep, ending with real entropy encoding/decoding.

```bash
python run_flame_steak.py     # N3DV
python run_banana.py          # HyperNeRF
```

Available scripts: `run_cook_spinach.py`, `run_cut_roasted_beef.py`, `run_flame_salmon_1.py`,
`run_flame_steak.py`, `run_sear_steak.py` (N3DV) and `run_3dprinter.py`, `run_banana.py`,
`run_broom2.py`, `run_chicken.py` (HyperNeRF).

Rate points used in each sweep:

| Dataset   | Rate points (`lmbda`)              | `lmbda_mask`   |
|-----------|------------------------------------|----------------|
| N3DV      | 0.002, 0.004, 0.008, 0.016         | `lmbda * 0.8`  |
| HyperNeRF | 0.001, 0.002, 0.004, 0.008, 0.016  | `lmbda`        |

### Illumination compensation (IC-4DGS)

The IC-4DGS setting ([paper](https://ieeexplore.ieee.org/document/11630311)) adds a lightweight
per-view illumination compensation module (`scene/ic_modules.py`) that is applied to the rendered image of
each training camera before the loss, so that photometric variations across cameras are absorbed by the
module instead of the Gaussians. A small L1 penalty on the predicted residual keeps the compensation
close to identity.

It is disabled by default. To enable it, add `--use_ic_module` to `COMMON` in the corresponding
`run_<scene>.py`, e.g.

```python
COMMON = (f"--eval --lod 0 --voxel_size 0.0333 --update_init_factor 128 --cam_mask 1 --use_ic_module "
          f"-s {DATA} --project_name release_{SCENE} --no_comet")
```

The IC module is currently supported on N3DV only; it is ignored (with a warning) for HyperNeRF scenes.

## Outputs

* Checkpoints and renderings: `LOG_ROOT/<scene>/base` and `LOG_ROOT/<scene>/l<lmbda>` (default `LOG_ROOT = ./output/release`).
* Metric summaries (`<name>.txt`, `<name>_comp.txt`): `output/release_<scene>/`.
  `<name>_comp.txt` reports the results after actual encoding/decoding (real bitstream size).

## Logging (optional)

Comet logging is disabled by the run scripts (`--no_comet`).
To enable it, fill in your `api_key` / `workspace` in `train.py` and drop `--no_comet`.


## Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{ho2025ted4dgstemporallyactivatedembeddingbased,
      title={TED-4DGS: Temporally Activated and Embedding-based Deformation for 4DGS Compression}, 
      author={Cheng-Yuan Ho and He-Bi Yang and Jui-Chiu Chiang and Yu-Lun Liu and Wen-Hsiao Peng},
      year={2025},
      eprint={2512.05446},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.05446}, 
}
```

If you use the illumination compensation setting, please also cite IC-4DGS:

```bibtex
@INPROCEEDINGS{11630311,
  author={Yang, He-Bi and He, Ming-Zhe and Yang, Cheng-Wei and Chiang, Jui-Chiu and Liu, Yu-Lun and Peng, Wen-Hsiao},
  booktitle={2026 IEEE International Conference on Image Processing (ICIP)}, 
  title={IC-4DGS: Illumination-Compensated 4D Gaussian Splatting Under Photometric Variations}, 
  year={2026},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ICIP61757.2026.11630311}}
```

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgement

 - We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) for presenting such an excellent work.
 - We thank all authors from [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for presenting such an excellent work.
 - We thank all authors from [HAC](https://github.com/YihangChen-ee/HAC) for presenting such an excellent work.
 - We thank all authors from [4DGaussians](https://github.com/hustvl/4DGaussians) for presenting such an excellent work.
