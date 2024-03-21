# Context-Aware Optimal Transport Learning for Retinal Fundus Image Enhancement (under review in MICCAI 2024)
Retinal fundus photography offers a non-invasive way to diagnose and monitor a variety of retinal diseases, but is prone to inherent quality glitches arising from systemic imperfections or operator/patient-related factors. However, high-quality retinal images are crucial for carrying out accurate diagnoses and automated analyses. The fundus image enhancement is typically formulated as a distribution alignment problem, by finding a one-to-one mapping between a low-quality image and its high-quality counterpart. This paper proposes a context-informed optimal transport (OT) learning framework for tackling unpaired fundus image enhancement. In contrast to standard generative image enhancement methods, which struggle with handling contextual information  (e.g., over-tampered local structures and unwanted artifacts), the proposed context-aware OT learning paradigm better preserves local structures and minimizes unwanted artifacts. Leveraging deep contextual features, we derive the proposed context-aware OT using the earth mover's distance, and show that the proposed context-OT has a solid theoretical guarantee.  
Experimental results on a large-scale dataset demonstrate the superiority of the proposed method over several state-of-the-art supervised and unsupervised methods in terms of signal-to-noise ratio, structural similarity index, as well as two downstream tasks. By enhancing image quality and performance in downstream tasks, the proposed method shows potential for advancing the utility of retinal fundus image-driven pipelines in routine clinical practice. 

## Data Pre-Processing 

To train our model, we used the pubicly available EyeQ dataset, which can be downloaded from [here](https://www.kaggle.com/c/diabetic-retinopathy-detection). We adapted the degradation technique mentioned in "Modeling and Enhancing Low-quality Retinal Fundus Images" [IEEE TMI, 2021]. Code for degradation is available [here](https://github.com/HzFu/EyeQ?tab=readme-ov-file).
## Pictorial representation of our work
![Contextual OT figure](https://github.com/Retinal-Research/Contextual-OT/assets/58003228/00eab6c6-0a15-493d-bff4-ff0f63476bac)

## Installing modules for Contextual loss

This work is inspired from "The Contextual Loss for Image Transformation with Non-Aligned Data" [ECCV 2018].Please clone the [repo](https://github.com/S-aiueo32/contextual_loss_pytorch) to use the pretrained VGG-19 to extract contextual embeddings. 

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

- [Git](https://git-scm.com)
- [Python](https://www.python.org/downloads/) and [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (optional)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Retinal-Research/Contextual-OT.git

2. Create a Python Environment and install the required libraries by running
   ```sh
   pip install -r requirements.txt

### Inference using Pre-trained weights

### Training our model.  

## Visual results

![image](https://github.com/Retinal-Research/Contextual-OT/assets/58003228/35fea2ab-2701-46c9-b149-96b6b0151957)

## Downstream Task results

![image](https://github.com/Retinal-Research/Contextual-OT/assets/58003228/6f766852-ac4b-4962-9cf3-b69bf210a6c3)
