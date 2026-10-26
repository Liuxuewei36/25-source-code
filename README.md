# BraTS and KiTS Segmentation Projects

Source code and saved experiment outputs for two medical image segmentation projects:

- **BraTS2019**: multimodal MRI brain tumor segmentation using a 3D U-Net. Source: `brats/源码/`; saved outputs: `brats/输出/`.
- **KiTS19**: kidney and tumor segmentation using a 2.5D U-Net baseline. Source: `kits/source code/`; saved outputs: `kits/输出/`; training records: `kits/训练记录/`.

## Setup

Install dependencies with `python -m pip install -r requirements.txt`.

Before training or evaluation, update the dataset paths in each project's `config.py` to match your local dataset location. The original datasets must be obtained separately. Run scripts from the corresponding source directory.

Saved checkpoints, evaluation artifacts, figures, and training records are included as archived experiment outputs.
