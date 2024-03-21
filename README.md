# Context-Aware Optimal Transport Learning for Retinal Fundus Image Enhancement (under review in MICCAI 2024)
Retinal fundus photography offers a non-invasive way to diagnose and monitor a variety of retinal diseases, but is prone to inherent quality glitches arising from systemic imperfections or operator/patient-related factors. However, high-quality retinal images are crucial for carrying out accurate diagnoses and automated analyses. The fundus image enhancement is typically formulated as a distribution alignment problem, by finding a one-to-one mapping between a low-quality image and its high-quality counterpart. This paper proposes a context-informed optimal transport (OT) learning framework for tackling unpaired fundus image enhancement. In contrast to standard generative image enhancement methods, which struggle with handling contextual information  (e.g., over-tampered local structures and unwanted artifacts), the proposed context-aware OT learning paradigm better preserves local structures and minimizes unwanted artifacts. Leveraging deep contextual features, we derive the proposed context-aware OT using the earth mover's distance, and show that the proposed context-OT has a solid theoretical guarantee.  
Experimental results on a large-scale dataset demonstrate the superiority of the proposed method over several state-of-the-art supervised and unsupervised methods in terms of signal-to-noise ratio, structural similarity index, as well as two downstream tasks. By enhancing image quality and performance in downstream tasks, the proposed method shows potential for advancing the utility of retinal fundus image-driven pipelines in routine clinical practice. 

## Pictorial representation of our work
![image](https://github.com/Retinal-Research/Contextual-OT/assets/58003228/899a2edb-bb51-4d98-a083-8ce9b6df6659)


### Installing modules for Contextual loss

This work is inspired from The Contextual Loss for Image Transformation with Non-Aligned Data [ECCV 2018].Please clone the repo https://github.com/S-aiueo32/contextual_loss_pytorch to use the pretrained VGG-19 to extract contxtual embeddings. 

#### Visual results

![image](https://github.com/Retinal-Research/Contextual-OT/assets/58003228/35fea2ab-2701-46c9-b149-96b6b0151957)

### Downstream Task results

![image](https://github.com/Retinal-Research/Contextual-OT/assets/58003228/6f766852-ac4b-4962-9cf3-b69bf210a6c3)
