# Pneumonia Detection from Chest X-Rays

## Project Overview

In this project, we will analyze data from the NIH Chest X-ray Dataset and train a CNN to classify a given chest x-ray for the presence or absence of pneumonia. This project will culminate in a model that can predict the presence of pneumonia with human radiologist-level accuracy.

Given medical images with clinical labels for each image that were extracted from their accompanying radiology reports. 

The project requires access to a GPU for fast training of deep learning architecture, as well as access to 112,000 chest x-rays with disease labels  acquired from 30,000 patients.

## Pneumonia and X-Rays : Overview

Chest X-ray exams are one of the most frequent and cost-effective types of medical imaging examinations. Deriving clinical diagnoses from chest X-rays can be challenging, however, even by skilled radiologists. 

When it comes to pneumonia, chest X-rays are the best available method for diagnosis. More than 1 million adults are hospitalized with pneumonia and around 50,000 die from the disease every
year in the US alone. The high prevalence of pneumonia makes it a good candidate for the development of a deep learning application for two reasons: 1) Data availability in a high enough quantity for training deep learning models for image classification 2) Opportunity for clinical aid by providing higher accuracy image reads of a difficult-to-diagnose disease and/or reduce clinical burnout by performing automated reads of very common scans. 

The diagnosis of pneumonia from chest X-rays is difficult for several reasons: 
1. The appearance of pneumonia in a chest X-ray can be very vague depending on the stage of the infection
2. Pneumonia often overlaps with other diagnoses
3. Pneumonia can mimic benign abnormalities

For these reasons, common methods of diagnostic validation performed in the clinical setting are to obtain sputum cultures to test for the presence of bacteria or viral bodies that cause pneumonia, reading the patient's clinical history and taking their demographic profile into account, and comparing a current image to prior chest X-rays for the same patient if they are available. 

## About the Dataset

The dataset provided to you for this project was curated by the NIH specifically to address the problem of a lack of large x-ray datasets with ground truth labels to be used in the creation of disease detection algorithms. 

The data can be downloaded from the [kaggle website](https://www.kaggle.com/nih-chest-xrays/data) and run it locally. You are STRONGLY recommended to use GPU to accelerate the training process,  since the data is huge.

There are 112,120 X-ray images with disease labels from 30,805 unique patients in this dataset.  The disease labels were created using Natural Language Processing (NLP) to mine the associated radiological reports. The labels include 14 common thoracic pathologies: 
- Atelectasis 
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia 

The biggest limitation of this dataset is that image labels were NLP-extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.

The original radiology reports are not publicly available but you can find more details on the labeling process [here.](https://arxiv.org/abs/1705.02315) 


### Dataset Contents: 

1. 112,120 frontal-view chest X-ray PNG images in 1024*1024 resolution (under images folder)
2. Meta data for all images (Data_Entry_2017.csv): Image Index, Finding Labels, Follow-up #,
Patient ID, Patient Age, Patient Gender, View Position, Original Image Size and Original Image
Pixel Spacing.


## Project Steps

### 1. Exploratory Data Analysis

The first part of this project will involve exploratory data analysis (EDA) to understand and describe the content and nature of the data.

Note that much of the work performed during your EDA will enable the completion of the final component of this project which is focused on documentation of your algorithm for the FDA. This is described in a later section, but some important things to focus on during your EDA may be: 

* The patient demographic data such as gender, age, patient position,etc. (as it is available)
* The x-ray views taken (i.e. view position)
* The number of cases including: 
    * number of pneumonia cases,
    * number of non-pneumonia cases
* The distribution of other diseases that are comorbid with pneumonia
* Number of disease per patient 
* Pixel-level assessments of the imaging data for healthy & disease states of interest (e.g. histograms of intensity values) and compare distributions across diseases.

### 2. Building and Training Your Model

**Training and validating Datasets**

From the findings in the EDA component of this project, the training and validation sets are preperly curated for classifying pneumonia. Following points are taken into consideration:

* Distribution of diseases other than pneumonia that are present in both datasets
* Demographic information, image view positions, and number of images per patient in each set
* Distribution of pneumonia-positive and pneumonia-negative cases in each dataset

**Model Architecture**

 VGG16 architecture is used with weights trained on the ImageNet dataset. Fine-tuning can be performed by freezing your chosen pre-built network and adding several new layers to the end to train, or by doing this in combination with selectively freezing and training some layers of the pre-trained network. 


**Image Pre-Processing and Augmentation** 

 Preprocessing is done prior to feeding images into the network for training and validating. This serves the purpose of conforming to the model's architecture and/or for the purposes of augmenting the training dataset for increasing the model performance.

**Training** 

while training the model, following parameters are tweaked to improve performance: 
* Image augmentation parameters
* Training batch size
* Training learning rate 
* Inclusion and parameters of specific layers in your model 

 **Performance Assessment**

Model is trained for several epocs and varied learning rate.'accuracy' meric is used for evaluation.

#### Final Result obtained

---
1/1 [==============================] - 1s 560ms/step - loss: 0.2853 - accuracy: 0.9286
1/1 [==============================] - 0s 278ms/step - loss: 0.4509 - accuracy: 0.8214
([0.28528881072998047, 0.9285714030265808],
 [0.45091989636421204, 0.8214285969734192])

---
Training time 10 epochs
Model 1 : Saved by best weight checkpoint at 4th epoch. [0.28528881072998047, 0.9285714030265808]
Model 2 : Final trained result [0.45091989636421204, 0.8214285969734192]

 __Note that detecting pneumonia is *hard* even for trained expert radiologists, so you should *not* expect to acheive sky-high performance.__ [This paper](https://arxiv.org/pdf/1711.05225.pdf) describes some human-reader-level F1 scores for detecting pneumonia, and can be used as a reference point for how well your model could perform.

### 3. Clinical Workflow Integration 

The imaging data provided for training your model was transformed from DICOM format into .png to help aid in the image pre-processing and model training steps of this project. In the real world, however, the pixel-level imaging data are contained inside of standard DICOM files. 
