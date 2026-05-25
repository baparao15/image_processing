# Crowd Counting on ShanghaiTech Dataset

This project builds a crowd counting pipeline for the ShanghaiTech dataset using deep learning and density-aware regression. The work is organized as a notebook-first workflow and covers:

- ground-truth count generation from density maps
- image preprocessing with RGB + CLAHE fusion
- a multi-model Part A pipeline for dense scenes
- a simpler Part B pipeline for lower-density scenes
- final CSV result generation for submission and analysis

## Project Overview

Crowd counting becomes difficult when people are very small, heavily overlapping, and spread unevenly across the image. Instead of treating the problem as normal object detection, this project uses convolutional neural networks to learn crowd texture, density patterns, and total count directly from images.

The dataset is split into two different problems:

- **ShanghaiTech Part A**: very dense scenes with large count variation
- **ShanghaiTech Part B**: comparatively lighter and more regular scenes

Because these two parts behave differently, they are handled using different model pipelines.

## What This Project Does

The full workflow is:

1. Read processed density maps and compute the actual count for each image.
2. Preprocess images into a fused 6-channel representation.
3. Train and evaluate crowd counting models for Part A and Part B.
4. Generate part-wise prediction CSV files.
5. Merge predictions into a final results CSV.

## Preprocessing

The preprocessing used in this project is one of the main ideas behind the model input.

For each image:

1. The image is loaded in **RGB** format.
2. Pixel values are normalized to the range `[0, 1]`.
3. The same image is converted to **grayscale**.
4. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) is applied to improve local contrast.
5. The enhanced grayscale image is repeated into 3 channels.
6. The original RGB image and the enhanced grayscale image are concatenated channel-wise.

This creates a **6-channel fused input**:

- channels 1 to 3: RGB
- channels 4 to 6: CLAHE-enhanced grayscale

This helps the model use both:

- normal scene appearance from RGB
- enhanced local structure and crowd texture from grayscale contrast enhancement

## Model Design

### CountNet

The core model family used in this project is a custom CNN-based architecture referred to in the notebooks as **CountNet**.

CountNet uses:

- convolutional layers for image feature extraction
- residual blocks for deeper and more stable learning
- squeeze-and-excitation attention for channel-wise feature weighting
- multiple output heads depending on the task

In the Part A workflow, the model learns more than one signal:

- a **density map head** to learn where crowd concentration exists
- a **count regression head** to predict the total count
- a **bucket/classification head** to predict coarse crowd level

This multi-head learning helps the model understand both spatial and global crowd information.

### Why CNN Is Used

CNNs are used because crowd images contain local spatial patterns such as:

- small head-like shapes
- repeated crowd textures
- dense clusters
- contrast differences between crowd and background

The convolution layers learn these patterns automatically from data instead of requiring hand-crafted rules.

## Part A Pipeline

Part A is the harder half of the dataset because it contains much denser scenes and larger count variation.

To handle that, this project uses **three separately trained models** built from the same CountNet architecture:

- **sparse model**
- **low model**
- **dense model**

These are not three different architectures. They are three different trained instances of the same CNN design. The main difference is that each model is trained on a different crowd-density regime, so the learned weights become specialized.

### Part A Inference Flow

For one Part A image:

1. The image is preprocessed into the fused 6-channel format.
2. The same input is passed to the sparse, low, and dense models.
3. Each model produces its own count-related prediction.
4. A guarded fusion/selection stage compares the outputs.
5. A final Part A count is produced.

In practice, the **low model** usually acts as the base predictor, while the **dense model** is allowed to influence the final prediction when the image appears truly crowded.

## Part B Pipeline

Part B is more stable and less dense than Part A, so the modeling strategy is simpler.

For Part B, the project uses:

- **one fused-input count model**

This means:

1. The image is preprocessed in the same RGB + CLAHE fused way.
2. The fused image is passed through one CNN-based model.
3. The model predicts the count directly.
4. Light calibration is used when needed.

So Part B uses a single-model pipeline instead of the three-model guarded pipeline used for Part A.

## Repository Structure

```text
C:\Users\pendy\Desktop\s6\projects\dip
├── DATASET
│   ├── actual_counts.csv
│   ├── prediction_results.csv
│   ├── prediction_results_part_a.csv
│   ├── prediction_results_part_b.csv
│   └── part_A / part_B images
├── notebooks
│   ├── 01_create_actual_counts_csv.ipynb
│   ├── 02_part_a_patch_density.ipynb
│   ├── 03_part_b_count_model.ipynb
│   ├── 04_create_prediction_results.ipynb
│   ├── 90_part_a_single_image_explainer.ipynb
│   └── 91_part_b_single_image_explainer.ipynb
├── processed_density_maps
│   └── part_A / part_B density map files (.h5)
├── tools
│   └── generate_explainer_notebooks.py
└── requirements.txt
```

## Notebook Guide

### [01_create_actual_counts_csv.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/01_create_actual_counts_csv.ipynb)

Creates the ground-truth counts by summing density maps and exports `actual_counts.csv`.

### [02_part_a_patch_density.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/02_part_a_patch_density.ipynb)

Main Part A notebook. Contains:

- fused preprocessing
- CountNet-based dense-scene modeling
- sparse / low / dense model workflow
- validation and final Part A prediction generation

### [03_part_b_count_model.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/03_part_b_count_model.ipynb)

Main Part B notebook. Contains:

- fused preprocessing
- single-model Part B training/inference
- count prediction and calibration flow

### [04_create_prediction_results.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/04_create_prediction_results.ipynb)

Combines part-wise outputs into the final result CSV and prints aggregate metrics.

### [90_part_a_single_image_explainer.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/90_part_a_single_image_explainer.ipynb)

Presentation/demo notebook that explains how one Part A image is processed step by step.

### [91_part_b_single_image_explainer.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/91_part_b_single_image_explainer.ipynb)

Presentation/demo notebook that explains how one Part B image is processed step by step.

## Installation

Create and activate a Python environment, then install dependencies.

```powershell
py -m venv dip
.\dip\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m pip install h5py nbformat
```

Minimal dependencies currently listed in [requirements.txt](C:/Users/pendy/Desktop/s6/projects/dip/requirements.txt):

- `ultralytics`
- `opencv-python`
- `numpy`
- `requests`

Depending on your environment, you may also need:

- `torch`
- `torchvision`
- `pandas`
- `matplotlib`
- `h5py`
- `jupyter`
- `nbformat`

## How To Run

Open Jupyter Notebook or JupyterLab inside the project folder:

```powershell
.\dip\Scripts\jupyter-lab.exe
```

Recommended execution order:

1. Run [01_create_actual_counts_csv.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/01_create_actual_counts_csv.ipynb)
2. Run [02_part_a_patch_density.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/02_part_a_patch_density.ipynb)
3. Run [03_part_b_count_model.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/03_part_b_count_model.ipynb)
4. Run [04_create_prediction_results.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/04_create_prediction_results.ipynb)

For viva/demo explanation on a single sample image:

1. Run [90_part_a_single_image_explainer.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/90_part_a_single_image_explainer.ipynb)
2. Run [91_part_b_single_image_explainer.ipynb](C:/Users/pendy/Desktop/s6/projects/dip/notebooks/91_part_b_single_image_explainer.ipynb)

## Outputs

Important output files in [DATASET](C:/Users/pendy/Desktop/s6/projects/dip/DATASET):

- [actual_counts.csv](C:/Users/pendy/Desktop/s6/projects/dip/DATASET/actual_counts.csv)
- [prediction_results_part_a.csv](C:/Users/pendy/Desktop/s6/projects/dip/DATASET/prediction_results_part_a.csv)
- [prediction_results_part_b.csv](C:/Users/pendy/Desktop/s6/projects/dip/DATASET/prediction_results_part_b.csv)
- [prediction_results.csv](C:/Users/pendy/Desktop/s6/projects/dip/DATASET/prediction_results.csv)

These files store:

- actual count
- predicted count
- signed error
- absolute error
- part-wise model information

## Presentation / Viva Summary

If you need a short explanation in presentation form:

- preprocessing uses RGB + CLAHE grayscale fusion to create a 6-channel input
- Part A uses three CountNet models: sparse, low, dense
- Part B uses one fused-input model
- CountNet is a CNN-based model with residual blocks and attention
- final predictions are exported to CSV for evaluation

## Notes

- Part A is substantially harder than Part B because of dense scenes and large count variation.
- The explainer notebooks were added to make the model flow easier to present image-by-image.
- This repository is notebook-centric, so the main implementation lives inside the notebooks rather than a packaged Python module.
#
