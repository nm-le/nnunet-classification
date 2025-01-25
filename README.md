# Deep Learning for Automatic Pancreas Cancer Segmentation and Classification in 3D CT Scans

Accurate pancreas cancer segmentation and subtype classification in 3D CT scans is a crucial step in computer-aided diagnosis and treatment planning. In this project, we present a **multi-task deep learning** approach that simultaneously performs:
1. **Pancreas and lesion segmentation**, and  
2. **Lesion subtype classification**.

Building upon the [nnU-Net framework], we incorporate a classification branch into the shared encoder–decoder architecture, taking inspiration from the [PANDA approach]. Our method processes each volumetric CT scan in a unified pipeline, generating both segmentation maps and subtype predictions.


---

## Table of Contents

1. [Background & Related Work](#background--related-work)
2. [Methods](#methods)
   - [Data Exploration](#data-exploration)
   - [Dataset and DataLoader Modifications](#dataset-and-dataloader-modifications)
   - [Architecture Modifications](#architecture-modifications)
   - [Pipeline Modifications](#pipeline-modifications)
   - [Training Configuration](#training-configuration)
3. [Results](#results)
4. [Discussion](#discussion)
5. [References](#references)

---

## Background & Related Work

**nnU-Net**  
- A prominent segmentation model in the medical imaging domain, featuring out-of-the-box image preprocessing and a model training pipeline.  
- Uses an encoder-decoder framework with skip connections to learn deep representations from medical scans.  
- Well-known for its **plug-and-play** nature and has been adapted as a **backbone** for tasks beyond segmentation.

**PANDA**  
- Extends nnU-Net for **pancreas classification**.  
- Specifically, Stage 2 of PANDA trains an nnU-Net backbone as a joint segmentation and classification network to detect *Pancreatic ductal adenocarcinoma (PDAC)*.  
- This is done by attaching global max-pooling and a fully connected layer to each decoder stage, concatenating these features, and training the network with a combination of segmentation and classification losses.

### Our Objective

- **Replicate** the design and results of the PANDA framework.
- **Explore** alternative architectural modifications of nnU-Net for multi-task learning (joint segmentation and classification).  
- **Dataset**: A small set of 3D CT scans of pancreatic cancer patients (252 scans).  
- **Tasks**:
  1. Segment each scan into **3 regions**: background, pancreas, lesion.  
  2. Classify each scan into **one of three labeled subtypes**.

---

## Methods

### Data Exploration

We used a **training dataset** of 252 3D CT scans of the pancreas region. Each scan is labeled with one of **3 subtypes**, and has a corresponding **segmentation mask**.  


> **Observation**: The number of samples per subtype is **imbalanced**, skewed particularly for Subtype 1.  

To address this imbalance, we tested two strategies:
1. **Weighted Cross-Entropy Loss** for classification.  
2. **Modified data sampling** in batch generation to ensure each class appears more uniformly.

> We did not apply additional custom augmentations or transformations; instead, we used nnU-Net’s **built-in** augmentations (via `batchgenerators`).

---

### Dataset and DataLoader Modifications

**Relevant Files**  
- `nnUNet/nnunetv2/training/dataloading/data_loader_3d.py`  
- `nnUNet/nnunetv2/training/dataloading/nnunet_dataset.py`  

To incorporate classification:
1. **nnUNetDataset**:  
   - The dataset class constructs a dictionary with case names as keys and metadata (including segmentation info) as values.  
   - We **added** each case’s *class label* to this dictionary so it is accessible during training.

2. **nnUNetDataLoader3D**:  
   - We need to **expose** class labels to the training loop.  
   - Because of **class imbalance**, we **modified the sampling** process to load classes in a more balanced manner across an epoch.

---

### Architecture Modifications

**Relevant Files**  
- `dynamic-network-architectures/dynamic_network_architectures/building_blocks/unet_decoder.py`  
- `dynamic-network-architectures/dynamic_network_architectures/architectures/unet.py`  

We added a **classification branch** to nnU-Net’s encoder–decoder architecture, inspired by the PANDA approach. We explored two approaches:

1. **Single-Stage Classification Head**  
   - Apply a *global max-pooling* + *fully-connected* (linear) layer **only** to the **last decoder stage**.

2. **Multi-Stage Classification Heads** (PANDA-style)  
   - Apply a *global max-pooling* + *fully-connected* layer to **each** decoder stage, **concatenate** them, and feed that to a final classifier.

Additionally, since our classification has **3 classes**, we experimented with a **one-vs-rest** (OvR) strategy, training three binary classifiers and combining results for the final prediction.

---

### Pipeline Modifications

**Relevant Files**  
- `nnUNet/nnunetv2/training/nnUNetTrainer/variants/classification/nnUNetTrainerMultiTask.py`  
- `nnUNet/nnunetv2/inference/predict_from_raw_data.py`  

Because we now receive **segmentation outputs** *and* **classification logits** from the network:

1. **Training**  
   - We combine the standard **segmentation loss** (`L_seg`) with a **classification loss** (`L_cls`) in a single objective:  
     \[
       \mathcal{L} = \mathcal{L}_{seg} + 0.3 \cdot \mathcal{L}_{cls}.
     \]  
   - This coefficient (0.3) follows the PANDA strategy to balance segmentation vs. classification optimization.

2. **Inference**  
   - During inference, the network outputs both **segmentation maps** and **classification logits**.  
   - We implemented a mechanism to **save** classification logits and predicted classes to a text file for post-processing.

---

### Training Configuration

- **Batch size**: 8  
- **Epochs**: 100  
- **Number of training samples**: 252  

---

## Results

We evaluate the network using:
- **Dice Similarity Coefficient (DSC)** for segmentation.  
- **Area Under the ROC Curve (AUC)** for classification (one-vs-rest).

Below are the results for our **best-performing** configuration:  
- **Architecture**: PANDA-style (multi-stage heads)  
- **Batch size**: 8  

### Segmentation Results
**Average DSC**: **0.8999**

### Classification Results (One-vs-Rest AUC)


---

## Discussion

Our **multi-task network** achieved a strong segmentation performance (**average DSC ~ 0.8999**), comparable to other nnU-Net baselines. However, classification results, especially for **class 0**, have **room for improvement**.

1. **Data**  
   - **Limited dataset size** (252 scans) can lead to overfitting and poor generalization.  
   - **Class imbalance** skews the model toward the majority class. Despite balanced sampling, underrepresented classes still pose a challenge.

2. **Architecture**  
   - The PANDA-style **multi-stage feature concatenation** yielded the best overall performance.  
   - Future investigations could explore **attention mechanisms** or **multi-head attention** to better leverage encoder features for classification.

3. **Loss Function & Regularization**  
   - We combined segmentation and classification losses with a **0.3** weight on the classification term (per PANDA).  
   - Tuning this weight or adding more **regularization** (e.g., dropout, weight decay) might further **improve classification** without degrading segmentation.

---

## References
1. ["nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation."](https://github.com/MIC-DKFZ/nnUNet)  
2. ["Large-scale pancreatic cancer detection via non-contrast CT and deep learning."](https://www.nature.com/articles/s41591-023-02640-w)  
3. ["Metrics reloaded: recommendations for image analysis validation."](https://www.nature.com/articles/s41592-023-02151-z)


