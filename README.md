# TikUDA

**Efficient Unsupervised Domain Adaptation Regression for Spatial-Temporal Sensor Fusion**

[**IEEE Internet of Things Journal**](https://doi.org/10.1109/JIOT.2025.3597150) · [**arXiv**](https://arxiv.org/abs/2411.06917)

Official code repository for the paper by **Keivan Faghih Niresi, Ismail Nejjar, and Olga Fink**, published in *IEEE Internet of Things Journal*, vol. 12, no. 20, pp. 42559–42572, 2025.

## Overview

TikUDA is an unsupervised domain adaptation method for regression with spatial-temporal sensor data. It learns from a labeled source domain while adapting to an unlabeled target domain. The method combines supervised regression with feature alignment derived from Tikhonov-regularized Gram matrices.

The alignment objective has two components:

- **Angle alignment** 
- **Scale alignment** 

For the sensor-fusion experiments, the feature extractor consists of a linear encoder, gated recurrent units (GRUs) for temporal modeling, and a graph attention network (GAT) for spatial modeling. A regression head produces the target estimate.


## Repository structure

```text
TikUDA/
├── data/
│   ├── R-212.csv            # Air-quality measurements from route R-212
│   └── R-69.csv             # Air-quality measurements from route R-69
├── losses.py               # TikUDA and baseline alignment losses
├── main.py                 # Training and evaluation entry point
├── model.py                # GRU-GAT feature extractor and regression model
├── training_testing.py     # Training and inference routines
└── utils.py                # Preprocessing, sequence construction, and seeding
```

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/EPFL-IMOS/TikUDA.git
cd TikUDA
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install PyTorch and the dependencies used by the code:

```bash
python -m pip install torch numpy pandas scikit-learn
python -m pip install torch-geometric torch-spatiotemporal
```

The model imports `torch_geometric` and `tsl` (Torch Spatiotemporal). Consult the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and the [Torch Spatiotemporal repository](https://github.com/TorchSpatiotemporal/tsl) if your PyTorch, CUDA, or operating-system versions require a particular installation procedure. An environment lockfile is not provided, so compatible dependency versions may need to be selected for your system.

## Data

The two CSV files in `data/` contain air-quality measurements from **R-212** and **R-69**. Their columns include reference ozone and nitrogen-dioxide measurements, low-cost sensor readings, temperature, and humidity.

The public preprocessing code:

1. Reads the semicolon-separated CSV files and removes the `date` column.
2. Uses the first remaining column (`ref_o3`) as the regression target and the other columns as model inputs.
3. Applies min-max scaling fitted on the source domain.
4. Constructs sequences with a window length of **16** time steps.

The target domain's input sequences are used for unsupervised adaptation; target labels are used for evaluation rather than for optimizing the adaptation loss. To run a different prediction target, such as `ref_no2`, update the column selection in `utils.py` accordingly.

## Usage

For adaptation from **R-212 to R-69**:

```bash
python main.py \
  --source_dataset R-212.csv \
  --target_dataset R-69.csv \
  --device cpu \
  --seed 42
```

For the reverse direction, **R-69 to R-212**:

```bash
python main.py \
  --source_dataset R-69.csv \
  --target_dataset R-212.csv \
  --device cpu \
  --seed 42
```

The script uses `Fast_TikUDA` by default and reports RMSE and MAE in normalized and original target units. The default batch size is 64 and the learning rate is `3e-4`. Change `--device` to a valid CUDA device (for example, `cuda:0`) if GPU execution is available.

### Notes on the current entry point

Before running the current repository version, check these two lines in the source code:

- In `main.py`, the training loop currently reads `for e in range(10)`. To use the configured number of epochs (150 by default), change it to `for e in range(args.epochs)`.


## Citation

If you use TikUDA in your research, please cite:

```bibtex
@article{niresi2025tikuda,
  author  = {Keivan Faghih Niresi and Ismail Nejjar and Olga Fink},
  title   = {Efficient Unsupervised Domain Adaptation Regression for Spatial-Temporal Sensor Fusion},
  journal = {IEEE Internet of Things Journal},
  volume  = {12},
  number  = {20},
  pages   = {42559--42572},
  year    = {2025},
  doi     = {10.1109/JIOT.2025.3597150}
}
```


