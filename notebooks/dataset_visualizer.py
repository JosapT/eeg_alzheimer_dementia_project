import mne
import matplotlib.pyplot as plt

eeg_file_raw = r"/Users/josephthi/Desktop/UCI_Classes/cs184A/CS_184A_Final Project/data/ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
spectogram_file = r"/Users/josephthi/Desktop/UCI_Classes/cs184A/CS_184A_Final Project/data/spectrogram/sub-001_spe.npy"

# Load raw EEG
def display_raw_data(eeg_file_raw):
    raw_eeg = mne.io.read_raw_eeglab(eeg_file_raw, preload=True)

    # Print metadata
    print(raw_eeg.info)

    # Plot EEG
    raw_eeg.plot(n_channels=30, duration=10, scalings='auto')
    plt.show()


def display_spectogram_data(data):
    plt.specgram(data[:][0][0], Fs=128, cmap='jet_r')
    plt.show()

display_raw_data(eeg_file_raw)


