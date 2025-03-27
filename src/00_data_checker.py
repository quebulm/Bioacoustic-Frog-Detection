import os
import librosa
import numpy as np


# ------------------------------
# Load Audio Data
# ------------------------------

def load_audio(file_path, sr=None):
    """
    Loads an audio file and returns the waveform and sample rate.

    Args:
        file_path (str): Path to the audio file.
        sr (int, optional): Sampling rate. If None, keeps original.

    Returns:
        tuple: (audio waveform as np.array, sample rate)
    """
    try:
        data, sr = librosa.load(file_path, sr=sr)
        return data, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


# ------------------------------
# Load Labels from TXT Files
# ------------------------------

def load_labels(label_folder, audio_folder):
    """
    Loads all TXT annotation files and finds matching WAV files.

    Args:
        label_folder (str): Path to the label folder containing TXT files.
        audio_folder (str): Path to the audio folder containing WAV files.

    Returns:
        dict: Mapping { "audio_filename.wav": [ (start_time, end_time) ] }
    """
    labels = {}

    for txt_file in os.listdir(label_folder):
        if txt_file.endswith(".Table.1.selections.txt"):
            txt_path = os.path.join(label_folder, txt_file)

            # Extract corresponding WAV filename
            audio_file = txt_file.replace(".Table.1.selections.txt", ".wav")
            audio_path = os.path.join(audio_folder, audio_file)

            # Only process if the matching WAV file exists
            if not os.path.exists(audio_path):
                print(f"Skipping {txt_file} (No matching WAV file)")
                continue

            try:
                with open(txt_path, "r") as f:
                    lines = f.readlines()

                # Extract relevant lines (skip header)
                time_intervals = []
                for line in lines[1:]:  # Skip the header line
                    parts = line.strip().split("\t")  # Tab-separated columns
                    if len(parts) < 5:
                        continue  # Skip malformed lines

                    start_time = float(parts[3])  # "Begin Time (s)"
                    end_time = float(parts[4])  # "End Time (s)"
                    time_intervals.append((start_time, end_time))

                # Save intervals only if we have valid labels
                if time_intervals:
                    labels[audio_file] = time_intervals

            except Exception as e:
                print(f"Error loading {txt_path}: {e}")

    return labels


# ------------------------------
# Load Entire Dataset (Only Labeled Files)
# ------------------------------

def load_dataset(audio_folder, label_folder, sr=None):
    """
    Loads only labeled WAV files and corresponding labels into a dataset.

    Args:
        audio_folder (str): Path to the folder containing raw WAV files.
        label_folder (str): Path to the folder containing TXT annotations.
        sr (int, optional): Sampling rate. If None, keeps original.

    Returns:
        list: A list of tuples (audio_waveform, sr, label_data)
    """
    labels = load_labels(label_folder, audio_folder)
    dataset = []

    for audio_file, label_data in labels.items():  # Only iterate over labeled files
        file_path = os.path.join(audio_folder, audio_file)

        # Load audio
        audio_waveform, sample_rate = load_audio(file_path, sr=sr)
        if audio_waveform is None:
            continue  # Skip if loading failed

        # Append to dataset
        dataset.append((audio_waveform, sample_rate, label_data))

    return dataset


# ------------------------------
# Dataset statistics
# ------------------------------

if __name__ == "__main__":
    # Define data paths
    audio_folder = "../data/Raw"
    label_folder = "../data/Labels"

    # Load dataset
    dataset = load_dataset(audio_folder, label_folder, sr=44100)

    # Print dataset statistics
    print(f"Loaded {len(dataset)} labeled audio files.")

    print(f"\n3 Example Files:")
    for i, (audio, sr, labels) in enumerate(dataset[:3]):
        print(f"File {i + 1}: {len(audio)} samples, {sr} Hz, Labels: {labels}")