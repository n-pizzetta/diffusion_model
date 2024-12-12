# Diffusion Model with U-Net on CelebA Dataset

## Introduction
This report delves into the theoretical and practical aspects of diffusion models and their integration with U-Net architectures for image generation tasks. We focus on implementing a score-based diffusion model on the CelebA dataset, demonstrating the process from understanding the underlying mathematics to generating high-quality face images.

## Background

### What Are Diffusion Models?
Diffusion models are a class of generative models that gradually destroy structure in a data distribution through a forward diffusion process and then learn to reverse this process to generate data samples from noise. They leverage a Markov chain of latent variables that starts from a simple distribution (e.g., Gaussian noise) and iteratively refines samples to match the target data distribution.

### Score-Based Diffusion
Score-based models estimate the gradient of the log-probability density (the "score") of the data. By training a neural network to predict the noise present in a noisy image, the model indirectly learns how to move samples toward the data manifold. Stochastic differential equations (SDEs) can generalize diffusion to continuous-time formulations, offering flexible sampling procedures.

### U-Net Architecture
The U-Net architecture, introduced for biomedical image segmentation, is a powerful encoder-decoder model with skip connections. The encoder compresses spatial information to learn global context, while the decoder reconstructs the image from a latent representation. Skip connections preserve high-frequency details, making U-Net ideal for reconstructive tasks such as denoising.

In diffusion models, U-Net acts as the core neural network predicting the added noise at each timestep. The skip connections help preserve structure and enable the model to generate coherent images from noise.

## Methodology

### Dataset: CelebA
- **CelebA**: A large-scale face attributes dataset containing over 200,000 face images.
- **Preprocessing**: 
  - Resized images to 64x64.
  - Normalized pixels to [-1, 1].
  
This standardized format ensures consistent input for the diffusion process and U-Net model.

### Forward Diffusion Process
The forward process adds noise to clean images over multiple timesteps. Let $x_0$ be the original image and $x_t$ the noisy image at timestep $t$. The forward diffusion uses a noise schedule to progressively increase the variance of the Gaussian noise:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
$$

Here, $\bar{\alpha}_t$ controls the amount of noise added at timestep $t$.

### Reverse Diffusion Process
The model learns to reverse the noising process. Given $x_t$, it predicts $\epsilon_\theta(x_t, t)$, the noise at timestep $t$. Using this prediction, we estimate $x_0$ and refine the sample backward through time:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}.
$$

Then we sample $x_{t-1}$ from the conditional distribution using the predicted noise.

### U-Net Integration
The U-Net model $ \epsilon_\theta $ takes as input the noisy image $x_t$ and the timestep $t$. The network is time-conditioned, which means it incorporates a time embedding vector into the latent space, enabling it to adapt its noise prediction to the current timestep.

**Architecture Details**:
- **Encoder Path**: Gradually downsample the image to a bottleneck layer, capturing broad semantic information.
- **Time Embedding**: A multi-layer perceptron (MLP) transforms the scalar timestep into a vector, then reshaped and added to intermediate features before the bottleneck.
- **Decoder Path**: Upsample the latent representation, combine it with encoder features via skip connections, and refine to produce a noise estimate with the same spatial dimensions as the input.

## Training Procedure

### Loss Function
We use a mean squared error (MSE) loss between the predicted noise $\hat{\epsilon}_\theta(x_t, t)$ and the actual noise $\epsilon$:

$$
\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right].
$$

Minimizing this loss trains the U-Net to accurately predict noise at each timestep.

### Hyperparameters
- **Batch Size**: 8
- **Image Size**: 64x64
- **Timesteps**: 500
- **Learning Rate**: 1e-4
- **Epochs**: 50 (adjusted based on computational resources and convergence)

These values were chosen as a starting point and may be tuned for improved results.

## Results (Based on the Google pre-trained model)

### Qualitative Analysis
- Initially generated images might appear blurry or lack detail.
- With sufficient training, the U-Net begins to recover facial features, hair structure, and facial attributes.
- The final samples resemble realistic human faces, though some artifacts or less realistic attributes may still appear, depending on model capacity and training duration.

### Quantitative Evaluation
- Compute metrics like the Fréchet Inception Distance (FID) or Inception Score (IS) to evaluate image quality.
- Due to the complexity of setup, we may rely initially on qualitative inspection. As an extension, integrate FID computations to track improvement.

## Challenges and Solutions

- **Training Stability**: Diffusion models are generally stable, but ensuring correct noise scheduling and proper normalization is crucial.
- **Computational Resources**: Training can be slow, especially with a large dataset and 500 timesteps. Using mixed precision (AMP) and multiple GPUs can help.
- **Hyperparameter Tuning**: Adjusting noise schedules, batch size, and learning rates can improve sample quality.

## Conclusions and Future Work

This project demonstrates:
- A working implementation of a diffusion model on the CelebA dataset.
- How U-Net architectures integrate effectively with diffusion processes.
- The capability of score-based models to produce realistic images.

**Future Directions**:
- **Higher Resolution**: Train models on 128x128 or larger images.
- **Improved Time Embeddings**: Experiment with sinusoidal embeddings or more sophisticated conditioning.
- **Conditional Generation**: Incorporate attribute labels from CelebA to control generated features.
- **State-of-the-Art Improvements**: Explore advanced techniques like improved noise schedules, DDIM sampling, or classifier-free guidance to enhance quality and diversity.

---

**References**:
- Ho, J., et al. "[Denoising Diffusion Probabilistic Models.](documentation/denoising-diffusion-proba-mod.pdf)" *NeurIPS 2020*.
- Song, Y., & Ermon, S. "[Score-Based Generative Modeling Through SDEs.](documentation/score-based_gen_mod_SDE.pdf)" *ICLR 2021*.
- Ronneberger, O., et al. "[U-Net: Convolutional Networks for Biomedical Image Segmentation.](documentation/u-net_paper.pdf)" *MICCAI 2015*.
