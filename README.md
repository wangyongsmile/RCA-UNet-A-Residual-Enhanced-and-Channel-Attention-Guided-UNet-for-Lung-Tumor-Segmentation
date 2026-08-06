# RCA-UNet: Residual-Enhanced and Channel-Attention-Guided U-Net for Lung Tumor Segmentation

## Description

This repository contains the PyTorch implementation used in the article **“RCA-UNet: A Residual-Enhanced and Channel-Attention Guided U-Net for Lung Tumor Segmentation.”**

RCA-UNet is a two-dimensional encoder-decoder segmentation network. Standard convolutional blocks in the encoder and decoder are replaced with residual modules, and channel-attention modules are inserted into the skip pathways to recalibrate encoder features before fusion.

The repository contains:

- RCA-UNet implementations for the LUNA16 and MSD experiments;
- implementations of the comparison models;
- component ablation experiments;
- data-augmentation, dropout, loss-weight, and learning-rate-scheduler ablations;
- preprocessing scripts;
- training code; and
- evaluation metrics.

## Repository

https://github.com/wangyongsmile/RCA-UNet-A-Residual-Enhanced-and-Channel-Attention-Guided-UNet-for-Lung-Tumor-Segmentation

## Dataset information

The datasets are **not redistributed in this repository**.

### LUNA16

- Dataset: LUng Nodule Analysis 2016 (LUNA16)
- Source: official LUNA16 challenge
- Underlying image collection: LIDC-IDRI, The Cancer Imaging Archive
- Modality: chest computed tomography
- Official LUNA16 download page: https://luna16.grand-challenge.org/Download/
- LUNA16 Zenodo records:
  - https://doi.org/10.5281/zenodo.2595812
  - https://doi.org/10.5281/zenodo.2596478
- LIDC-IDRI dataset DOI: https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
- LIDC-IDRI license: Creative Commons Attribution 3.0 Unported

Required LIDC-IDRI acknowledgement:

> The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.

### MSD Task06_Lung

- Dataset: Medical Segmentation Decathlon
- Task: Task06_Lung
- Modality: computed tomography
- Target: lung tumor
- Official dataset page: http://medicaldecathlon.com/
- Official AWS download page: https://medicaldecathlon.com/dataaws/
- License: Creative Commons Attribution-ShareAlike 4.0 International
- Dataset paper: Antonelli et al., *Nature Communications* (2022), “The Medical Segmentation Decathlon.”

The expected original MSD structure is:

```text
Task06_Lung/
├── imagesTr/
│   ├── lung_001.nii.gz
│   └── ...
├── labelsTr/
│   ├── lung_001.nii.gz
│   └── ...
└── dataset.json
```

## Code information

The current public repository contains two experiment branches:

```text
code/
├── LUNA/
│   ├── comparison_and_module_ablation/
│   └── parameter_ablation/
└── MSD/
    ├── comparison_and_module_ablation/
    ├── parameter_ablation/
    └── processed.py
```

## Requirements

- Python 3.10 or later
- PyTorch
- NumPy
- SciPy
- pandas
- scikit-learn
- scikit-image
- nibabel
- SimpleITK
- matplotlib
- tqdm
- albumentations
- OpenCV-Python
- Pillow
- torchvision

Install the dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For strict reproducibility, replace the minimum-version requirements with the exact versions exported from the environment used to generate the manuscript results:

```bash
pip freeze > requirements-lock.txt
```

## Methodology

### MSD preprocessing

1. Download `Task06_Lung`.
2. Open `code/MSD/processed.py`.
3. Set:
   - `DATA_PATH` to the directory containing `imagesTr` and `labelsTr`;
   - `SAVE_DIR` to the desired preprocessed-data directory.
4. Run:

```bash
python code/MSD/processed.py
```

The script:

- matches image and annotation files;
- uses random seed 42;
- performs a patient-level 8:2 training-validation split;
- extracts two-dimensional axial slices;
- resizes slices to 256 × 256 pixels;
- applies CT windowing and normalization;
- saves image and label arrays as `.npy` files.

**Important consistency check:** the current script removes slices with fewer than 50 tumor pixels from both training and validation subsets and currently uses linear interpolation for label masks. The manuscript and code must be made identical before resubmission.

### MSD model training

From the MSD comparison directory, train RCA-UNet with:

```bash
python main.py \
  --model rca_unet \
  --data-dir /path/to/LungPreprocessed_63 \
  --batch-size 8 \
  --epochs 120 \
  --lr 8e-5 \
  --weight-decay 2e-5 \
  --results-dir results
```

Available comparison and component-ablation model names are:

```text
rca_unet
att_unet
drs_cnn2
incremental_mrrn
r2_unet
basic_unet
segnet
no_residual_rca_unet
no_channel_attention_rca_unet
```

Example:

```bash
python main.py --model basic_unet --data-dir /path/to/LungPreprocessed_63
python main.py --model no_residual_rca_unet --data-dir /path/to/LungPreprocessed_63
python main.py --model no_channel_attention_rca_unet --data-dir /path/to/LungPreprocessed_63
```

### LUNA16 preprocessing and training

1. Download the LUNA16 CT subsets and annotations.
2. Edit `BASE_PATH` in the LUNA comparison experiment `config.py`.
3. Confirm that the expected CT and annotation paths match the loader implementation.
4. From the LUNA comparison directory, run:

```bash
python main.py
```

The released configuration sets:

```text
random seed: 42
batch size: 8
epochs: 200
learning rate: 2e-4
weight decay: 1e-4
```

The current LUNA `main.py` sequentially trains the comparison models and RCA-UNet variants.

## Ablation experiments

The repository contains scripts for:

- residual-module ablation;
- channel-attention ablation;
- no, moderate, and strong data augmentation;
- no, moderate, and high dropout;
- loss-weight comparisons; and
- learning-rate-scheduler comparisons.

Run each script from its corresponding `parameter_ablation` directory. Store outputs in clearly named subdirectories and retain the exact configuration used for every manuscript figure.

## Evaluation

The released code calculates:

- Dice coefficient;
- intersection over union;
- sensitivity;
- specificity;
- precision;
- 95th-percentile Hausdorff distance; and
- average surface distance.

The threshold for binary segmentation is 0.5.

**Distance-unit note:** the current implementation calculates HD95 and ASD in pixel units unless image spacing is explicitly supplied. Do not report these values as millimetres without spacing-aware calculations.


## Citation

Please cite the associated article after publication. The MSD and LIDC-IDRI datasets must also be cited according to their data-use policies.

## License

The source code is released under the MIT License. The third-party datasets retain their original licenses and are not covered by the code license.

## Contribution guidelines

Issues and pull requests are welcome. Contributions should:

- use English comments and documentation;
- follow PEP 8 where practical;
- avoid committing patient data, derived dataset files, model checkpoints containing restricted data, local absolute paths, or cache files;
- include a description of the change and a reproducible test command.
