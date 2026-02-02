# Robustness of VQA and Captioning Models with BLIP-2

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Project Description
This project focuses on evaluating the robustness of Visual Question Answering (VQA) and image captioning models, specifically utilizing the BLIP-2 model architecture from the LAVIS library. The goal is to analyze how these models perform under various conditions, including adversarial attacks, and to provide a framework for generating and evaluating such scenarios.

## 📋 Table of Contents
- [Setup and Installation](#-setup-and-installation)
- [VQA Robustness Evaluation Workflow](#-vqa-robustness-evaluation-workflow)
- [Captioning Robustness Evaluation Workflow](#-captioning-robustness-evaluation-workflow)
- [Adversarial Image Generation](#-adversarial-image-generation)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

## 🛠 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- Conda (recommended for environment management)

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/username/robustness_of_vqa_and_captioning_models.git
    cd robustness_of_vqa_and_captioning_models
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n blip2_robustness python=3.9
    conda activate blip2_robustness
    ```

3.  **Install Python dependencies:**
    The project relies on the LAVIS library. Install its dependencies directly from its `requirements.txt`.
    ```bash
    pip install -r LAVIS/requirements.txt
    pip install numpy torch pillow # Ensure these core dependencies are also installed if not covered by LAVIS
    ```

## 🔬 VQA Robustness Evaluation Workflow

This workflow outlines the steps to evaluate the robustness of the BLIP-2 model on VQA tasks.

### 1. Download VQA and COCO Datasets
*   **COCO Images:**
    Download the COCO 2014 validation images using the script from LAVIS. These images are required by the VQA dataset.
    ```bash
    python LAVIS/lavis/datasets/download_scripts/download_coco.py
    ```
    Ensure the `val2014` images are in `datasets/coco/val2014`.

*   **VQA v2 Annotations and Questions:**
    Download the VQA v2 validation questions (`v2_OpenEnded_mscoco_val2014_questions.json`) and annotations (`v2_mscoco_val2014_annotations.json`) from the official VQA website ([https://visualqa.org/](https://visualqa.org/)). Place these files in `datasets/vqa/`.

### 2. Create VQA Dataset Subset
Create a balanced subset of the VQA validation dataset.
```bash
python src/vqa_dataset_subset_creation.py
```
This script creates `vqa_subset_20_per_type.json` (you can configure the number of samples per type in the script).

### 3. Prepare Adversarial Images
Place your pre-generated adversarial images for the VQA task in the `vqa_adversarial_output/` directory. See the [Adversarial Image Generation](#-adversarial-image-generation) section for more details.

### 4. Run VQA Evaluation
Execute the evaluation script to assess the model's performance on the adversarial VQA subset.
```bash
python src/blip2_vqa_adversarial_evaluation.py
```
This script uses the generated subset and adversarial images, saving the results to `subset_evaluation_results.json`.

## ✍️ Captioning Robustness Evaluation Workflow

This workflow outlines the steps to evaluate the robustness of the BLIP-2 model on image captioning tasks.

### 1. Download COCO Dataset
Download the COCO 2014 validation images and captions.
```bash
python LAVIS/lavis/datasets/download_scripts/download_coco.py
```
Ensure the `val2014` images are in `datasets/coco/val2014` and the `captions_val2014.json` is available in the appropriate annotations directory.

### 2. Create Captioning Dataset Subset
Create a subset of COCO validation images and their annotations.
```bash
python src/caption_dataset_subset_creation.py
```
This will create `caption_subset_200_images.json` and a `caption_subset_images/` directory with the sampled images.

### 3. Prepare Adversarial Images
Place your pre-generated adversarial images for the captioning task in the `caption_adversarial_output/` directory. See the [Adversarial Image Generation](#-adversarial-image-generation) section for more details.

### 4. Run Captioning Evaluation
Execute the evaluation script to assess the model's captioning performance on original vs. adversarial images.
```bash
python src/blip2_caption_adversarial_evaluation.py
```
This script uses the generated subset, original images, and adversarial images, saving the results to `caption_evaluation_results.json`.

## ⚔️ Adversarial Image Generation

This repository primarily focuses on *evaluating* the robustness using pre-generated adversarial images. The provided evaluation scripts expect these adversarial images to be present in specific directories.

*   **Expected Directories:**
    *   `caption_adversarial_output/`: Contains adversarial images generated for captioning tasks.
    *   `vqa_adversarial_output/`: Contains adversarial images generated for VQA tasks.

The naming convention for adversarial images often follows `original_image_filename@target_image_filename.png`. If you wish to generate your own adversarial images, you would need to implement that process separately. This project provides the framework for their evaluation.

## 📈 Results

The evaluation scripts will generate JSON files containing detailed results and aggregated metrics.

*   `subset_evaluation_results.json`: Contains results from the VQA adversarial evaluation.
*   `caption_evaluation_results.json`: Contains results from the captioning adversarial evaluation.

These files include per-sample results, overall accuracy (for VQA), and various captioning metrics (ROUGE, BERTScore, CLIP Score, BLEU) for both original and adversarial images.

## 📁 Project Structure
```
.
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── caption_adversarial_output/  # Pre-generated adversarial images for captioning
├── caption_subset_images/       # Subset of COCO images for captioning evaluation
├── coco-caption/                # COCO Caption Evaluation Tool
├── configs/                     # Configuration files (model, dataset, experiment)
├── datasets/                    # Raw dataset files (COCO, VQA)
│   ├── coco/                    # COCO images and annotations
│   └── vqa/                     # VQA questions and annotations
├── docs/                        # Documentation
├── LAVIS/                       # Modified LAVIS library (git submodule/copy)
├── notebooks/                   # Jupyter notebooks for analysis
├── scripts/                     # Utility scripts
├── src/                         # Source code for data processing and evaluation
│   ├── blip2_caption_adversarial_evaluation.py
│   ├── blip2_vqa_adversarial_evaluation.py
│   ├── caption_dataset_subset_creation.py
│   ├── vqa_dataset_subset_creation.py
│   └── ... (other source files)
├── tests/                       # Unit tests
├── vqa_adversarial_output/      # Pre-generated adversarial images for VQA
└── vqa_subset_images/           # Subset of COCO images for VQA evaluation
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation
If you use this code in your research, please cite:

```bibtex
@misc{robustness_of_vqa_and_captioning_models,
  author = {Your Name}, # TODO: Replace with actual author names
  title = {Robustness of VQA and Captioning Models},
  year = {2024}, # TODO: Update year if necessary
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/username/robustness_of_vqa_and_captioning_models}} # TODO: Update GitHub URL
}
```

## 🙏 Acknowledgments
-   Acknowledgment 1
-   Acknowledgment 2
-   Acknowledgment 3