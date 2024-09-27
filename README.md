# Cancer molecular subtyping using limited multi-omics data with missingness
CancerSD is an end-to-end model designed for **Cancer** **S**ubtype **D**iagnosis using limited weakly-paired multi-omics data

## Introduction

Diagnosing cancer subtypes is a prerequisite for precise treatment. Existing multi-omics data fusion-based diagnostic solutions build on the requisite of sufficient samples with complete multi-omics data, which is challenging to obtain in clinical applications. To address the bottleneck of collecting sufficient samples with complete data in clinical applications, we proposed a flexible integrative model (CancerSD) to diagnose cancer subtype using limited samples with incomplete multi-omics data. CancerSD designs contrastive learning tasks and masking-and-reconstruction tasks to reliably impute missing omics, and fuses available omics data with the imputed ones to accurately diagnose cancer subtypes. To address the issue of limited clinical samples, it introduces a category-level contrastive loss to extend the meta-learning framework, effectively transferring knowledge from external datasets to pretrain the diagnostic model. Experiments on benchmark datasets show that CancerSD not only gives accurate diagnosis, but also maintains a high authenticity and good interpretability. In addition, CancerSD identifies important molecular characteristics associated with cancer subtypes, and it defines the Integrated CancerSD Score that can serve as an independent predictive factor for patient prognosis.

![The framework of CancerSD.](https://github.com/user-attachments/assets/9d9c0ce6-2e5d-4d2f-9432-82e6d94c6a43)

Our CancerSD has the advantage to accurately diagnose cancer subtypes by leveraging incomplete multi-omics data of patients and to reduce its dependence on the quantity of in-house cancer samples for training by absorbing knowledge from external datasets. The CancerSD pipeline comprises four components dedicated to accomplishing reliable and flexible cancer subtype diagnosis in scenarios characterized by incomplete data and scarce samples. 
(i) It firstly establishes the patient feature encoder, a tensor-based fusion network, to efficiently integrate multi-omics data from samples. 
(ii) Then, it constructs the missing omics imputation network to reliably impute missing omics of samples, which consists of an encoder, a projector, and multiple omics-specific generators. After that, it defines Contrastive Learning tasks alongside \revised{Masking-And-Reconstruction (MAR)} tasks to optimize this imputation network. The former explores the consistent patient representations across different augmented views, while the latter utilizes such representations to impute the missing omics data.
(iii) Next, it introduces the cancer subtype diagnosis network that fuses available and imputed omics data to calculate the probability of each patient suffered from a particular subtype. 
(iv) To enable model optimization on the scarce in-house clinical samples, CancerSD further proposes a knowledge transfer network to extract meta-knowledge from external datasets. 
We wanted to remark that the first three networks are collectively referred as CancerSD backbone or the base learner (CancerSD_b), while the last network is designated as the meta learner (CancerSD_m).

CancerSD is based on the Python program language. For the model training, We used an Nvidia GeForce RTX 3090 (24G) GPU that hosted on a Ubuntu 22.04.1 LTS machine with Intel Xeon Gold 6248R CPUs and 512GB memory. The version of CUDA is 11.7 and that of PyTorch is 1.13.1. When the GPU's memory is not enough to support the running of the tool, we suggest simplifying the network structure.

## Dependencies
* Python 3 >= 3.8
* PyTorch >= 1.12
* NumPy >= 1.23
* Pandas >= 1.5
* Scikit-learn >= 1.3

## Contents of this repository
In this repository, you can find the following folders:

## Usage

