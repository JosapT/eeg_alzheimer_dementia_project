import mne
import matplotlib.pyplot as plt

eeg_file_raw = r"/Users/josephthi/Desktop/UCI_Classes/cs184A/CS_184A_Final Project/ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

# Load raw EEG
raw_eeg = mne.io.read_raw_eeglab(eeg_file_raw, preload=True)

# Print metadata
print(raw_eeg.info)

# Plot EEG
raw_eeg.plot(n_channels=30, duration=10, scalings='auto')
plt.show()