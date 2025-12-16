# I'm Something of a Painter Myself - CycleGAN

## Competition Overview

This project is part of the Kaggle competition ["I’m Something of a Painter Myself"](https://www.kaggle.com/c/gan-getting-started). The goal of the competition is to build a Generative Adversarial Network (GAN) that generates Monet-style images from real photos. The evaluation metric is MiFID (Memorization-informed Fréchet Inception Distance), where lower scores indicate better quality and diversity of generated images.

## Project Structure

The repository is organized as follows:

- `src/models/`: Contains the GAN architectures (Generator and Discriminator).
- `notebooks/`: Jupyter notebooks corresponding to different experiments.
- `data/`: Dataset directories (Monet paintings and photos).
- `checkpoints/`: Saved model weights.

## Experiments & Architectures

We conducted multiple experiments to optimize the style transfer quality, focusing on different Generator architectures within the CycleGAN framework.

### Common Configurations

- **Framework**: CycleGAN (Unpaired Image-to-Image Translation).
- **Discriminator**: PatchGAN (used across all experiments).
  - **Type**: 70x70 PatchGAN classification.
  - **Structure**:
    - Series of 4x4 Convolutional layers (stride 2) to downsample the image.
    - Uses LeakyReLU (0.2) activations and Instance Normalization.
    - Final output layer produces a 1-channel feature map classifying local patches as real or fake.
- **Loss Functions**:
  - Adversarial Loss: MSE Loss (Least Squares GAN) for stable training.
  - Cycle Consistency Loss: L1 Loss (lambda=10) to preserve content.
  - Identity Loss: L1 Loss (lambda=5) to preserve color palette.
- **Data Processing**: Images resized to 256x256, normalized to [-1, 1].

### Experiment Details

#### Experiment 1 & 2: ResNet-based Generator

- **Architecture**: `ResNetGenerator`
- **Goal**: Establish a strong baseline using the standard CycleGAN architecture.
- **Technical Specs**:
  - **Encoder**:
    - 7x7 Conv (Reflection Pad), InstanceNorm, ReLU.
    - 2 Downsampling layers: 3x3 Conv (stride 2), InstanceNorm, ReLU.
  - **Bottleneck**:
    - 6 Residual Blocks (for 256x256 images).
    - Each block contains two 3x3 Convs with InstanceNorm and ReLU.
  - **Decoder**:
    - 2 Upsampling layers: 3x3 Transposed Conv (stride 2), InstanceNorm, ReLU.
    - Output layer: 7x7 Conv, Tanh activation.
  - **Filters**: Started with `ngf=32` filters.

#### Experiment 3 & 4: U-Net-based Generator

- **Architecture**: `UNetGenerator`
- **Goal**: Test if U-Net's skip connections improve the sharpness and detail retention of the generated paintings compared to ResNet.
- **Technical Specs**:
  - **Structure**: Encoder-Decoder network with skip connections between mirrored layers.
  - **Encoder (Down)**:
    - 4x4 Conv (stride 2), LeakyReLU (0.2).
    - Instance Normalization applied after the first layer.
  - **Decoder (Up)**:
    - 4x4 Transposed Conv (stride 2), ReLU.
    - Instance Normalization.
    - Concatenates output with the corresponding encoder feature map (Skip Connection).
    - Dropout (50%) used in some intermediate layers.
  - **Depth**: `num_downs=6` layers deep.
  - **Filters**: Used `ngf=32` (Exp 3) and `ngf=16` (Exp 4).

## Results

We achieved significant improvements by switching from ResNet to U-Net architectures.

### Performance Summary

| Experiment | Generator          | Score (MiFID) | Notes                                  |
| ---------- | ------------------ | ------------- | -------------------------------------- |
| **Exp 3**  | **U-Net (ngf=32)** | **77.55874**  | **Best Performance (65th Place)**      |
| Exp 4      | U-Net (ngf=16)     | 77.80592      | Very close to best, reduced parameters |
| Exp 2      | ResNet             | 85.47171      | Baseline performance (82nd Place)      |

### Leaderboard Proof

Below is the screenshot showing our scores across different experiments:

![Leaderboard Scores](fsdgbdg.png)

## "All Stuff" - Implementation Details

- **Training**: Models were trained on the provided dataset of Monet paintings (300 images) and photos (7000+ images).
- **Augmentation**: Basic resizing and normalization were applied. Some experiments explored additional augmentations but the reported scores utilize standard preprocessing.
- **Evaluation**: The models were evaluated using the MiFID metric on the Kaggle leaderboard.

## Usage

To reproduce the results, refer to the notebooks in the `notebooks/` directory.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training notebooks (e.g., `notebooks/experiment3.ipynb`) to train the models.
3. Use the submission notebooks to generate the final `images.zip` for Kaggle submission.
