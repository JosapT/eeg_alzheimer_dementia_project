# **Clustering + CNN for Alzheimers and Dementia Detection**
This project utilizes a combination of Clustering and Convolutional Neural Networks to classify people with Alzheimer's or Dementia using EEG data

# **Python Version**
Python 3.10.19

# **Libraries (BEFORE RUNNING)**
```
pip install -r requirements.txt
```
**Install with conda**
```
conda install --yes --file requirements.txt
```

# **Instructions**
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
