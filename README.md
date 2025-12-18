# I'm Something of a Painter Myself - CycleGAN

## Results

We achieved significant improvements by switching from ResNet to U-Net architectures and tuning augmentations.

### Performance Summary

| Experiment | Generator          | Parameters | Score (MiFID) | Notes                                                                                       |
| ---------- | ------------------ | ---------- | ------------- | ------------------------------------------------------------------------------------------- |
| **Exp 3**  | **U-Net (ngf=32)** | **~7.0M**  | **77.55874**  | **Best Performance (65th Place)**. Used augmentations.                                      |
| Exp 4      | U-Net (ngf=16)     | ~1.8M      | 77.80592      | Very close to best. **No augmentations used**.                                              |
| Exp 2      | ResNet             | ~1.9M      | 85.47171      | Baseline performance (82nd Place). Used augmentations. [https://wandb.ai/konstantine25b-free-university-of-tbilisi-/Monet_GAN_Eval_Exp2/reports/Experiment-2-Report-ResNet-Baseline--VmlldzoxNTQwNjUwNg?accessToken=c52i1s7z1ioqjl6j088uwusrkqvp9kqs0jj6rnnrwqi6181qr9i9uhy7tvwxxqo2](README_resnet_exp2.md) |

### Leaderboard Proof

Below is the screenshot showing our scores across different experiments:

![Leaderboard Scores](fsdgbdg.png)

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

### Experiment Details & Parameter Calculation

#### Experiment 1 & 2: ResNet-based Generator

- **Architecture**: `ResNetGenerator`
- **Parameter Count**: **~1.9 Million**
- **Calculation**:
  - The bulk of parameters comes from the **6 Residual Blocks** in the bottleneck.
  - Each block has two 3x3 convolution layers with 128 filters.
  - Approx: $6 \text{ blocks} \times 2 \text{ layers} \times (128 \times 128 \times 3 \times 3) \approx 1.77\text{M}$ params.
  - Encoder/Decoder layers add the remaining ~0.13M.
- **Goal**: Establish a strong baseline using the standard CycleGAN architecture.
- **Augmentation**: Used standard augmentations (Random Flip, Rotate, Crop).
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

#### Experiment 3: U-Net-based Generator (Best Score)

- **Architecture**: `UNetGenerator`
- **Parameter Count**: **~7.0 Million**
- **Calculation**:
  - U-Net is deeper and uses **Skip Connections**.
  - The skip connections concatenate encoder features with decoder features, doubling the input depth for upsampling layers.
  - A single upsampling layer (e.g., handling 512 input channels -> 256 output channels with 4x4 kernel) alone contributes $\approx 2\text{M}$ parameters ($512 \times 256 \times 4 \times 4$).
  - Summing all downsampling and concatenated upsampling layers yields the ~7.0M total.
- **Goal**: Test if U-Net's skip connections improve detail retention.
- **Augmentation**: Used standard augmentations (Random Flip, Rotate, Crop).
- **Technical Specs**:
  - **Structure**: Encoder-Decoder network with skip connections.
  - **Depth**: `num_downs=6` layers deep.
  - **Filters**: `ngf=32`.
  - **Key Feature**: Skip connections propagate low-level features directly to the decoder, preserving sharpness better than the ResNet bottleneck.

#### Experiment 4: U-Net-based Generator (No Augmentations)

- **Architecture**: `UNetGenerator`
- **Parameter Count**: **~1.8 Million**
- **Calculation**:
  - Same architecture as Exp 3, but **base filters (`ngf`) reduced from 32 to 16**.
  - Parameter count scales with the square of the filter reduction factor ($0.5^2 = 0.25$).
  - $7.0\text{M} \times 0.25 \approx 1.75\text{M}$.
- **Goal**: Test model performance without augmentations to isolate architecture benefits.
- **Augmentation**: **None** (Only Resize & Normalize).
- **Technical Specs**:
  - **Structure**: Same U-Net architecture as Exp 3 but with fewer filters.
  - **Filters**: `ngf=16` (Reduced capacity).
  - **Observation**: Even with 1/4 of the parameters of Exp 3 and no augmentation, this model performed exceptionally well, suggesting the U-Net inductive bias is well-suited for this task.

## "All Stuff" - Implementation Details

- **Training**: Models were trained on the provided dataset of Monet paintings (300 images) and photos (7000+ images).
- **Augmentation**:
  - **Exp 2 & 3**: Utilized data augmentation techniques to improve generalization.
  - **Exp 4**: No augmentations were applied to test the pure capability of the U-Net architecture.
- **Evaluation**: The models were evaluated using the MiFID metric on the Kaggle leaderboard.

## Usage

To reproduce the results, refer to the notebooks in the `notebooks/` directory.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training notebooks (e.g., `notebooks/experiment3.ipynb`) to train the models.
3. Use the submission notebooks to generate the final `images.zip` for Kaggle submission.
