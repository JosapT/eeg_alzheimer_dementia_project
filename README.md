# **Clustering + CNN for Alzheimers and Dementia Detection**
Clustering EEG to spectrogram data preprocessing with CNN training for each cluster

# **Python Version**
Python 3.10.19

# **Libraries**
**Install with conda**

```
conda install pytorch matplotlib numpy scipy scikit-learn mne
```

# **Running the code**
1. Open eeg_training_pipeline.py
2. Within ```if __name__ == "main":``` you put in the directory of where the clustered data is (sample data directory should already be there by default) and you can set an output directory where the checkpoints will be stored
3. Run the script using
```
python eeg_training_pipeline.py
```
