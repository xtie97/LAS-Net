# Longitudinally-aware Segmentation Network (LAS-Net) for Pediatric Hodgkin Lymphoma :bookmark_tabs:

This repository contains the code for the paper [**Automatic Quantification of Serial PET/CT Images for Pediatric Hodgkin Lymphoma Patients Using a Longitudinally-Aware Segmentation Network**](https://arxiv.org/abs/2404.08611) (under review). 

## Overview :mag_right:
**Summary**: 
A longitudinally-aware segmentation network (LAS-Net) trained on multi-center clinical trial data achieved high performance in automatic quantification of PET metrics for baseline and interim-therapy scans in pediatric Hodgkin lymphoma. 

**Key Points**:
- :chart_with_upwards_trend: LAS-Net leverages information from baseline PET images to inform and improve the analysis of interim PET, achieving an F1 score of 0.606 in detecting residual lymphoma, which was significantly better than models without longitudinal awareness (P<0.01).
- :trophy: When analyzing baseline PET/CT images, LAS-Net attained a mean Dice score of 0.772, demonstrating comparable performance to the best comparator method (P=0.32).
- :medal_sports: The quantitative PET metrics measured by LAS-Net, including qPET, ∆SUVmax, metabolic tumor volume (MTV) and total lesion glycolysis (TLG), were highly correlated with physician measurements, with Spearman’s correlations of 0.78, 0.80, 0.93 and 0.96, respectively. 

## Design Principle
We designed LAS-Net with a dual-branch architecture to accommodate baseline and interim PET/CT images, as illustrated below. One branch exclusively processes baseline PET (PET1) and predicts the corresponding lesion masks. The other branch focuses on interim PET (PET2), but also utilizes information extracted from the PET1 branch to generate masks of residual lymphoma. This architecture enables our model to gather useful information from PET1 to inform and improve the analysis of subsequent scans. Meanwhile, it ensures a one-way information flow, preventing PET2 information from influencing PET1 analysis. 

![LASNet Architecture](./images/lasnet_model.jpg)



## Usage 🚀

To run this project, you can use the pre-configured **Docker** container for easy setup. The Docker image is hosted on Docker Hub.

### Steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xtie97/LAS-Net.git
   ```
2. **Pull the Docker image from Docker Hub**:
    ```bash
   docker pull xtie97/xxt_radiomics
   ```
3. **Run the Docker container**:
   ```bash
   docker run -it --rm -v $(pwd):/workspace xtie97/xxt_radiomics
   ```



We already released our model weights in [**Dropbox**](https://www.dropbox.com/scl/fo/6ihu7tjk2yqe75bylyy0t/h?rlkey=sbuaip5qy0ep6mukcne9nwlxe&dl=0).




## Citation 📚



