import os
import librosa
import librosa.display
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor


# ------------------------------
# Bandpass Filter Implementation
# ------------------------------

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Creates a bandpass filter using the Butterworth filter.

    Args:
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (int): Sampling rate of the audio.
        order (int): Filter order (default=5).

    Returns:
        tuple: Filter coefficients (b, a).
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a bandpass filter to the input audio signal.

    Args:
        data (np.ndarray): Audio signal.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (int): Sampling rate.
        order (int): Filter order (default=5).

    Returns:
        np.ndarray: Filtered audio signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


# ------------------------------
# Noise Reduction using Spectral Gating
# ------------------------------

def noise_reduction(data, sr):
    """
    Reduces background noise using spectral gating.

    Args:
        data (np.ndarray): Audio signal.
        sr (int): Sampling rate.

    Returns:
        np.ndarray: Noise-reduced audio signal.
    """
    try:
        # Compute Short-Time Fourier Transform (STFT)
        stft = librosa.stft(data)
        magnitude, phase = librosa.magphase(stft)

        # Estimate noise floor using median over quiet frequencies
        noise_floor = np.median(magnitude, axis=1, keepdims=True)

        # Apply noise threshold (1.5 times the noise floor)
        threshold = noise_floor * 1.5
        magnitude_cleaned = np.where(magnitude > threshold, magnitude, 0)

        # Reconstruct the signal with cleaned magnitude
        stft_cleaned = magnitude_cleaned * phase
        return librosa.istft(stft_cleaned)

    except Exception as e:
        print(f"Noise reduction failed: {e}")
        return data  # Return original data if an error occurs


# ------------------------------
# Processing Single WAV File
# ------------------------------

def process_single_wav(file_path, output_folder, lowcut=500, highcut=6000):
    """
    Processes a single WAV file by applying bandpass filtering and noise reduction.

    Args:
        file_path (str): Path to the input WAV file.
        output_folder (str): Path to save the processed file.
        lowcut (float): Lower cutoff frequency (default=500 Hz).
        highcut (float): Upper cutoff frequency (default=6000 Hz).

    Returns:
        None
    """
    try:
        # Load the WAV file (keep original sampling rate)
        data, sr = librosa.load(file_path, sr=None)

        # Apply bandpass filtering
        data_filtered = bandpass_filter(data, lowcut, highcut, sr)

        # Apply noise reduction
        data_cleaned = noise_reduction(data_filtered, sr)

        # Save the processed audio file
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, os.path.basename(file_path))
        sf.write(output_file, data_cleaned, sr)
        print(f"Processed and saved: {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# ------------------------------
# Processing All WAV Files with Multithreading
# ------------------------------

def process_wav_files_multithreaded(data_folder, output_folder, lowcut=500, highcut=6000, max_threads=6):
    """
    Processes multiple WAV files in parallel using multithreading.

    Args:
        data_folder (str): Directory containing raw WAV files.
        output_folder (str): Directory to save processed files.
        lowcut (float): Lower cutoff frequency (default=500 Hz).
        highcut (float): Upper cutoff frequency (default=6000 Hz).
        max_threads (int): Number of threads for parallel processing (default=6).

    Returns:
        None
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Collect all WAV files in the input folder
    wav_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".wav")]

    # Process files in parallel
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [
            executor.submit(process_single_wav, wav_file, output_folder, lowcut, highcut)
            for wav_file in wav_files
        ]
        for future in futures:
            future.result()  # Ensure all threads complete execution


# ------------------------------
# Execution
# ------------------------------

if __name__ == "__main__":
    # Define input and output directories
    data_folder = "/Volumes/QuebSSD/bioacoustic_data/Data_for_inference/Data/Songmeter_1"
    output_folder = "/Volumes/QuebSSD/bioacoustic_data/Data_for_inference/Data/processed_wav_files_Songmeter_1"

    # Start processing
    process_wav_files_multithreaded(data_folder, output_folder)