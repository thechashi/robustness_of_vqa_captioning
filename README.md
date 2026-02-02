# Robustness of VQA and Captioning Models with BLIP-2

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Project Description
This project focuses on evaluating the robustness of Visual Question Answering (VQA) and image captioning models, specifically utilizing the BLIP-2 model architecture from the LAVIS library. The goal is to analyze how these models perform under various conditions, including adversarial attacks, and to provide a framework for generating and evaluating such scenarios.

## 📋 Table of Contents
- [Setup and Installation](#-setup-and-installation)
- [Data Preparation](#-data-preparation)
- [Adversarial Image Generation](#-adversarial-image-generation)
- [Model Evaluation](#-model-evaluation)
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

## 🗃 Data Preparation

This section outlines how to set up the necessary datasets for evaluation.

### 1. Download Raw Datasets

*   **COCO Dataset (Images and Captions):**
    The COCO 2014 validation split images are required. You can download and extract them using the provided script from the LAVIS library:
    ```bash
    python LAVIS/lavis/datasets/download_scripts/download_coco.py
    ```
    This script will download `train2014.zip`, `val2014.zip`, and `test2014.zip` (or `test2015.zip`) and extract them. The images for evaluation (e.g., `val2014`) are expected to be located in `datasets/coco/val2014` relative to the project root.

*   **VQA v2 Dataset (Questions and Annotations):**
    The VQA v2 dataset questions and annotations are required.
    -   Download the VQA v2 validation questions: `v2_OpenEnded_mscoco_val2014_questions.json`
    -   Download the VQA v2 validation annotations: `v2_mscoco_val2014_annotations.json`
    You can typically find these on the official VQA website ([https://visualqa.org/](https://visualqa.org/)).
    Place these JSON files into a directory named `datasets/vqa/` relative to the project root.
    *(Example path: `robustness_of_vqa_and_captioning_models/datasets/vqa/v2_OpenEnded_mscoco_val2014_questions.json`)*

### 2. Create Dataset Subsets

To facilitate focused evaluation, smaller subsets of the full datasets are used.

*   **For Captioning Evaluation:**
    This script creates a subset of COCO validation images and their annotations.
    ```bash
    python src/caption_dataset_subset_creation.py
    ```
    By default, this creates `caption_subset_200_images.json` and copies the selected images into `caption_subset_images/` within the project root. You can modify the `n_samples` variable in the script to change the number of images.

*   **For VQA Evaluation:**
    This script creates a balanced subset of the VQA validation dataset, sampling from the top 10 question types.
    ```bash
    python src/vqa_dataset_subset_creation.py
    ```
    By default, this creates `vqa_subset_20_per_type.json` in the project root. You can modify the `n_samples` variable in the script to change the number of samples per question type. The images for this subset are referenced from the `datasets/coco/val2014` directory.

## ⚔️ Adversarial Image Generation

This repository primarily focuses on *evaluating* the robustness using pre-generated adversarial images. The provided evaluation scripts expect these adversarial images to be present in specific directories.

*   **Expected Directories:**
    *   `caption_adversarial_output/`: Contains adversarial images generated for captioning tasks.
    *   `vqa_adversarial_output/`: Contains adversarial images generated for VQA tasks.

The naming convention for adversarial images often follows `original_image_filename@target_image_filename.png`. If you wish to generate your own adversarial images, you would need to implement that process separately. This project provides the framework for their evaluation.

## 📊 Model Evaluation

Once the datasets and adversarial images are prepared, you can run the evaluation scripts. These scripts use the `Salesforce/blip2-opt-2.7b` model and other relevant libraries (e.g., CLIP) which will be automatically downloaded by the `transformers` and `clip` libraries upon first run.

*   **BLIP-2 VQA Adversarial Evaluation:**
    This script evaluates the BLIP-2 model's performance on VQA tasks using the generated adversarial images.
    ```bash
    python src/blip2_vqa_adversarial_evaluation.py
    ```
    The script uses `vqa_subset_20_per_type.json` (or whichever subset you generated) and images from `vqa_adversarial_output/` (and potentially `vqa_subset_images/` for original images). It saves detailed results to `subset_evaluation_results.json` in the project root.

*   **BLIP-2 Captioning Adversarial Evaluation:**
    This script evaluates the BLIP-2 model's captioning ability on original and adversarial images.
    ```bash
    python src/blip2_caption_adversarial_evaluation.py
    ```
    The script uses `caption_subset_200_images.json` (or whichever subset you generated), original images from `caption_subset_images/`, and adversarial images from `caption_adversarial_output/`. It saves detailed results to `caption_evaluation_results.json` in the project root.

## 📈 Results

The evaluation scripts will generate JSON files containing detailed results and aggregated metrics.

*   `subset_evaluation_results.json`: Contains results from the VQA adversarial evaluation.
*   `caption_evaluation_results.json`: Contains results from the captioning adversarial evaluation.

These files include per-sample results, overall accuracy (for VQA), and various captioning metrics (ROUGE, BERTScore, CLIP Score, BLEU) for both original and adversarial images.

## 📁 Project Structure
```
.
├── .gitignore
├── blip2_evaluation_results.json
├── blip2_vqa_evaluation_results.json
├── caption_adv_image_gen.log
├── caption_evaluation_results.json
├── caption_subset_200_images.json
├── README.md
├── requirements.txt
├── setup.py
├── vqa_adv_image_gen.log
├── vqa_adversarial_image_gen.log
├── vqa_evaluation_results.json
├── vqa_subset_20_per_type.json
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
-   Acknowledgment 3# robustness_of_vqa_captioning
