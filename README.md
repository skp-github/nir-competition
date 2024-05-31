# Near-Infrared Colorization using Pix2Pix GAN-based Architecture

## Introduction

This repository contains the implementation of **Near-Infrared Colorization** using a Pix2Pix GAN-based architecture. The solution was developed as part of a competition to effectively colorize near-infrared images.

## Repository Structure

- **Code**: Contains the implementation of the Pix2Pix GAN-based architecture for colorization.
- **Data**: Includes sample data files used for training and testing.
- **Models**: Pre-trained models and checkpoints.
- **Docs**: Detailed documentation and references related to the project.

## Technology and Methodology

Near-infrared (NIR) imaging captures information beyond the visible spectrum, making it useful for various applications, including remote sensing, medical imaging, and surveillance. Colorizing NIR images can enhance their interpretability and usability. This project employs a Pix2Pix GAN-based architecture to learn the mapping from NIR to RGB images.

## Features

- **Pix2Pix GAN Architecture**: Utilizes a conditional generative adversarial network for image-to-image translation.
- **High-Quality Colorization**: Achieves realistic colorization of near-infrared images.
- **Pre-trained Models**: Includes pre-trained models for immediate use and further fine-tuning.

## Requirements

- Python 3.x
- TensorFlow or PyTorch (depending on implementation)
- NumPy
- OpenCV
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/skp-github/nir-competition.git
    cd nir-competition
    ```

## Usage

### Training the Model

To train the Pix2Pix GAN model on your dataset, use:
```bash
python colorization_project/training.py 
