# Diffusion Models Image Generation

This repository demonstrates how to implement and train diffusion models for image generation, using CelebA as a reference dataset. It includes custom UNet and diffusion model implementations, configurations, training scripts, experiments, and supporting documentation. The project structure is designed to separate configurations, data, code, experiments, results, and documentation in a clear and organized manner.

<p align="center">
  <img src="./results/generated_images/steps_comparison.png" width=100%>
</p>

## Project Structure

- **configs**  
  Contains YAML configuration files defining hyperparameters and other settings for training and generation. You can easily adjust parameters like image size, number of timesteps, learning rate, batch size, and more.

- **data**  
  Contains datasets and related CSV files. For CelebA, partition and attribute files are included here. The `processed` subdirectory stores preprocessed `.npy` image files ready for training, while `raw` holds the original, unprocessed images. You can find the CelebA dataset on [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

- **documentation**  
  Contains research papers, notes, and additional references used to understand the theoretical background of diffusion models, UNet architectures, and related deep learning concepts.

- **experiments**  
  A sandbox directory where smaller scale tests were performed. Here, you’ll find:
  - A simple UNet architecture trial.
  - A minimal diffusion model trained on dummy data (random noise).
  
  These experiments helped in understanding the fundamental mechanics of diffusion models and verifying that the training loop and data pipelines work correctly before scaling up.

- **image_gen.py**  
  A standalone script that uses pre-trained UNet and diffusion models to generate new images. This script demonstrates how to load a trained model, run the reverse diffusion process, and produce high-quality synthetic images.

- **models**  
  A directory for storing trained model checkpoints. This includes:
  - The diffusion model weights (`.pth`) files.
  - The UNet model weights that serve as the backbone for noise prediction.

- **results**  
  Stores images, logs, and other outputs produced after running training and generation. It includes:
  - `generated_images`: Final synthesized images from the pre-trained diffusion model.
  - `images_custom_network`: Outputs from the custom-trained diffusion models and UNets, allowing comparison with the pre-trained versions. As expected, the results from the custom network are not as refined or realistic as those from the pre-trained models. This discrepancy likely stems from the custom model having been trained on a much smaller subset of data, fewer epochs, or with less refined hyperparameters, leading to reduced image quality and less coherent features. Over time, with more data, longer training, and careful tuning, these results should improve and approach the quality seen in the pre-trained model’s outputs.

- **src**  
  The main codebase for the custom diffusion and UNet implementations, preprocessing, and training scripts. Key files include:
  - `diffusion.py`: Diffusion logic, forward and reverse noising steps, and sampling.
  - `unet.py`: Custom UNet architecture used for noise prediction.
  - `preprocess.py`: Preprocessing pipeline for raw images into normalized `.npy` files.
  - `train_diffusion.py`: Training script for the diffusion model.
  - `train_unet.py`: Training script for the custom UNet model.
  - `generate.py`: Script to generate images from the trained custom model.

## Getting Started

1. **Install Dependencies:**  
   Make sure you have Python and the required libraries. Install additional dependencies (such as `accelerate`, `diffusers`, `torch`, `numpy`, etc.) as indicated in `requirements.txt`.

2. **Prepare Data:**  
   Place your raw images in `data/raw` and run `preprocess.py` to convert and normalize them into `.npy` tensors in `data/processed`.

3. **Configure and Train:**  
   Adjust hyperparameters in `configs/config.yaml` as needed. Then, run `train_diffusion.py` or `train_unet.py` from `src` to begin training. Check `notebooks` for interactive experimentation.

4. **Generate Images:**  
   Once training is complete and model weights are stored in `models/`, use `generate.py` to generate new images from the custom model. Whereas in `image_gen.py` the script takes the Google pre-trained diffusion model and UNet to produce synthesized faces or other images depending on the dataset. You can find the models on [Hugging Face](https://huggingface.co/google/ddpm-celebahq-256).

5. **Explore Results:**  
   Inspect the outputs in `results/` to see how the quality of generated images improves over training or compare different model configurations and checkpoints.

## Additional Notes

- The `experiments` directory is not part of the main training pipeline but provides valuable insights into the internals of UNets and diffusion models through simplified experiments.
- The `documentation` folder and references therein offer theoretical grounding, ensuring that the implementation aligns with established research on diffusion models.

This repository serves as a comprehensive template for anyone looking to understand and implement diffusion models for image synthesis, from raw preprocessing to final image generation.