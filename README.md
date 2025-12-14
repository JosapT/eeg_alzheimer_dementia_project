# **Clustering + CNN for Alzheimers and Dementia Detection**

## Problem Overview
Alzheimer’s Disease is a brain disorder characterized by impairment and decline of memory and cognitive skills. Electroencephalography (EEG) is a test that records the electrical activity of the brain which aids in diagnosing Alzheimer’s. To detect Alzheimer’s, patients typically have their EEG measurements taken when they lay down in a closed eye resting state. This provides a baseline measurement of brain function, and it can be standardized across patients. However, EEG is susceptible to artifacts, which are unwanted recorded signals from the body, measurement equipment, or the external environment. Additionally, the mental state and thoughts of a patient change the EEG recordings in unwanted ways. These factors complicate EEG analysis by modifying or adding noise to the actual signal. 

## Proposed Methodology
To address these challenges, our proposed methodology utilizes clustering to categorize hidden patterns in the signal, ideally including those that come from artifacts or variations in the patient’s state of mind. The objective is to focus on the inherent differences between Alzheimer's and healthy patients as opposed to the unwanted fluctuations. After clustering, a classifier is trained on each group to determine if the patient exhibits signs of Alzheimer’s. To evaluate a new datapoint, it would first be assigned to the most appropriate cluster, and then the point is passed through the cluster’s classifier to identify if the patient has Alzheimer’s or not. 

## Dataset
**Dataset Source:** https://openneuro.org/datasets/ds004504/versions/1.0.7

**Access:** The dataset can be downloaded using Git Annex


## **Python Version**
**Python 3.10.19**

## **Libraries (INSTALL BEFORE RUNNING)**
It is recommened to use a virtual environment like conda or venv to manage dependencies.
```
pip install -r requirements.txt
```
**Installation if using conda**

You can add --name (name) after create to change the name environment
```
conda env create -f environment.yml
```

## **Demo Instructions**
1. Clone the repository
```
git clone https://github.com/JosapT/eeg_alzheimer_dementia_project
```
2. Open eeg_training_pipeline.py
3. Within ```if __name__ == "main":``` you put in the directory of where the clustered data is (sample data directory should already be there by default) and you can set an output directory where the checkpoints will be stored
3. Run the script using
```
python eeg_training_pipeline.py
```
