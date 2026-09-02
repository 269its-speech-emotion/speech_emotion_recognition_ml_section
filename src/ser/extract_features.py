################################################################################
#                                                                              #
#                   Feature Extraction Component for SER Project               #
#                   Extracts MFCC and MFCCT features from preprocessed audio   #
#                   for emotion recognition using CNN-based methods.           #
#                                                                              #
################################################################################

import librosa
import numpy as np
from scipy.stats import mode
from pathlib import Path

from tqdm import tqdm

from ser.config import logger
from ser.constants import (
    EMOTION_TO_INDEX,
    MFCCT_BIN_SIZE,
    N_MFCC,
    N_MELS,
    SAMPLE_RATE,
)


def pad_feature_sequences(
    feature_sequences: list[np.ndarray],
    padding_value: float = 0.0,
) -> np.ndarray:
    """
    Pad feature arrays to the same number of time steps.

    Each input array must have shape:
        (time_steps, feature_count)

    Returns:
        Array with shape:
        (number_of_samples, max_time_steps, feature_count)
    """
    if not feature_sequences:
        return np.empty((0, 0, 0), dtype=np.float32)

    if any(feature.ndim != 2 for feature in feature_sequences):
        raise ValueError(
            "Each feature array must have shape (time_steps, feature_count)"
        )

    feature_count = feature_sequences[0].shape[1]

    if any(feature.shape[1] != feature_count for feature in feature_sequences):
        raise ValueError(
            "All feature arrays must have the same number of features"
        )

    max_time_steps = max(
        feature.shape[0]
        for feature in feature_sequences
    )

    padded_features = np.full(
        (
            len(feature_sequences),
            max_time_steps,
            feature_count,
        ),
        fill_value=padding_value,
        dtype=np.result_type(*feature_sequences, np.float32),
    )

    for sample_index, feature in enumerate(feature_sequences):
        time_steps = feature.shape[0]
        padded_features[sample_index, :time_steps, :] = feature

    return padded_features



def extract_mfcc_features(
    audio_signal, 
    sample_rate=SAMPLE_RATE, 
    frame_length=25, 
    frame_step=10, 
    n_mfcc=N_MFCC, 
    n_mels=N_MELS
) -> np.ndarray:
    """Extract MFCC features from audio, per the paper's Section 3.3(a).

    Args:
        audio_signal (np.ndarray): Preprocessed audio (shape: n_samples).
        sample_rate (int): Audio sample rate (default: 16000 Hz).
        frame_length (int): Frame length in ms (default: 25 ms).
        frame_step (int): Frame step in ms (default: 10 ms).
        n_mfcc (int): Number of MFCC coefficients (default: 13).
        n_mels (int): Number of Mel filter banks (default: 26).

    Returns:
        np.ndarray: MFCC features (shape: 1, n_frames, n_mfcc).
    """
    # Convert frame params to samples
    hop_length = int(sample_rate * frame_step / 1000)  # Step: 10 ms = 160 samples
    win_length = int(sample_rate * frame_length / 1000)  # Length: 25 ms = 400 samples
    
    # Check signal length for framing
    if len(audio_signal) < win_length:
        raise ValueError(f"Signal too short: {len(audio_signal)} < {win_length}")
    
    mfcc = librosa.feature.mfcc(
        y=audio_signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        window='hamming',
    )
    
    logger.info(f"MFCC shape: {mfcc.shape}")
    
    # Transpose MFCCs to -> Frames as rows, coefficients as columns
    mfcc_features = mfcc.T
    logger.info(f"MFCC shape after transpose: {mfcc_features.shape}")
    
    # Add a new axis to match expected input shape for CNN: (1, n_frames, n_mfcc)
    final_mfcc_features = mfcc_features[np.newaxis, :, :]
    logger.info(f"Final MFCC shape: {final_mfcc_features.shape}")

    return final_mfcc_features


def compute_mode_or_fallback(bin_data):
    """Compute mode of MFCC bin data, falling back to mean if mode fails.

    Args:
        bin_data (np.ndarray): MFCC segment (shape: (bin_size, 1)).

    Returns:
        float: Mode or mean value.
    """
    try:
        mode_value = mode(bin_data, axis=0, keepdims=False).mode[0].item()
    except Exception as e:
        logger.warning(f"Mode error for shape {bin_data.shape}: {str(e)}. Using mean: {np.mean(bin_data, axis=0).item()}")
        mode_value = np.mean(bin_data, axis=0).item()
    return mode_value


def compute_time_domain_features(bin_data):
    """Compute 12 time-domain features for an MFCC bin.

    Args:
        bin_data (np.ndarray): MFCC segment (shape: (bin_size, 1)).

    Returns:
        np.ndarray: 12 features: MIN, MAX, Mean, Median, Mode, STD, VAR, COV, RMS, Q1, Q2, Q3.
    """
    # Init array for features
    features = np.zeros(12)

    # Compute features
    features[0] = np.min(bin_data, axis=0).item()  # MIN
    features[1] = np.max(bin_data, axis=0).item()  # MAX
    features[2] = np.mean(bin_data, axis=0).item()  # Mean
    features[3] = np.median(bin_data, axis=0).item()  # Median
    features[4] = compute_mode_or_fallback(bin_data)  # Mode
    features[5] = np.std(bin_data, axis=0).item()  # STD
    features[6] = np.var(bin_data, axis=0).item()  # VAR
    
    mean_value = np.mean(bin_data).item()
    std_value = np.std(bin_data).item()
    features[7] = (std_value / abs(mean_value) if mean_value != 0 else 0) # COV
    
    features[8] = np.sqrt(np.mean(bin_data ** 2, axis=0)).item()  # RMS
    features[9] = np.percentile(bin_data, 25, axis=0).item()  # Q1
    features[10] = np.percentile(bin_data, 50, axis=0).item()  # Q2
    features[11] = np.percentile(bin_data, 75, axis=0).item()  # Q3

    # Log feature summary
    logger.debug(f"Computed features for shape {bin_data.shape}: {features}")

    return features


def extract_mfcct_features(
    mfcc_features: np.ndarray,
    bin_size: int = MFCCT_BIN_SIZE,
    time_domain_features: int = 12,
) -> np.ndarray:
    """Extract MFCCT features by binning MFCCs and computing time-domain stats.

    Args:
        mfcc_features (np.ndarray): MFCC data (shape: batch_size, n_frames, n_mfcc).
        bin_size (int): Rows per bin (default: MFCCT_BIN_SIZE).
        time_domain_features (int): Number of features per bin (default: 12).

    Returns:
        np.ndarray: MFCCT features (shape: batch_size, n_bins * 12, n_mfcc).
    """
    # Validate input
    if len(mfcc_features.shape) != 3:
        raise ValueError(f"Expected 3D MFCC features, got {mfcc_features.shape}")

    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")
    
    if time_domain_features != 12:
        raise ValueError(f"time_domain_features must be 12, got {time_domain_features}")
    
    # Log initial shapes
    batch_size, n_frames, n_mfcc = mfcc_features.shape
    logger.info(f"Processing batch: size={batch_size}, frames={n_frames}, mfcc={n_mfcc}")

    # Initialize list to hold MFCCT features for each batch
    mfcct_batch = []

    for batch_idx in range(batch_size):
        current_mfcc = mfcc_features[batch_idx]
        
        current_bin_size = min(bin_size, n_frames)
        n_bins = max(1, n_frames // current_bin_size)
        
        current_mfcc = current_mfcc[:n_bins * current_bin_size, :]
        
        # Init Master Feature Vector (MFV)
        mfv = np.zeros(
            (n_bins * time_domain_features, n_mfcc),
            dtype=mfcc_features.dtype, 
        )
        logger.info(f"MFV shape: {mfv.shape} for batch {batch_idx}")

        # Process each MFCC coefficient
        for coefficient_index in range(n_mfcc):
            for bin_index in range(n_bins):
                start = bin_index * current_bin_size
                end = start + current_bin_size 
                
                bin_data = current_mfcc[
                    start: end,
                    coefficient_index: coefficient_index + 1
                ]
                
                features = compute_time_domain_features(bin_data)
                
                start_feature = bin_index * time_domain_features
                end_feature = start_feature + time_domain_features 
                
                mfv[start_feature:end_feature, coefficient_index] = features
    
        mfcct_batch.append(mfv)
    
    return np.stack(mfcct_batch, axis=0)


def extract_dataset_features_and_labels(organized_audio_data_folder: Path) -> dict:
    """
    Extracts features and labels from the organized audio dataset.

    Args:
        organized_audio_data_folder (Path): Path to the organized audio dataset folder.

    Returns:
        dict: A dictionary containing the features (X) and labels (Y).
    """
    # Implementation for extracting features and labels
    # This function should iterate through the organized dataset, preprocess audio files,
    # extract MFCC and MFCCT features, and return them along with their corresponding labels
    # Placeholder implementation
    
    X_MFCC = []
    X_MFCCT = []
    Y = []
    
    dataset_features_and_labels: dict = {}
    
    # Iterate through each emotion's folder in the organized audio raw dataset
    for emotion_folder in organized_audio_data_folder.iterdir():
        if emotion_folder.is_dir():  # Ensure it's a directory
            # Progress bar for files in the current folder
            for file in tqdm(emotion_folder.iterdir(), desc=f'Extracting features from {emotion_folder.name}'):
                
                # Load and preprocess the audio file
                if file.suffix.lower() not in {".wav", ".mp3"}:
                    continue
                audio_signal, sample_rate = librosa.load(file, sr=SAMPLE_RATE)
                
                # Extract MFCC features
                logger.info(f"Extracting MFCC features for file: {file.name}")
                mfcc_features = extract_mfcc_features(audio_signal, sample_rate)
                
                # Extract MFCCT features
                logger.info(f"Extracting MFCCT features for file: {file.name}")
                mfcct_features = extract_mfcct_features(mfcc_features, bin_size=MFCCT_BIN_SIZE)
                
                # Append features and corresponding label
                logger.info(f"Appending features and label for file: {file.name}, label: {emotion_folder.name}")
                
                X_MFCC.append(mfcc_features[0])  # Assuming mfcc_features shape is (1, n_frames, n_mfcc)
                X_MFCCT.append(mfcct_features[0])  # Assuming mfcct_features shape is (1, n_bins * 12, n_mfcc)
                Y.append(EMOTION_TO_INDEX[emotion_folder.name])  # Assuming folder name is the label
    
    padded_X_MFCC = pad_feature_sequences(X_MFCC)
    padded_X_MFCCT = pad_feature_sequences(X_MFCCT)

    logger.info(f"Total samples processed: {len(Y)}")
    logger.info(f"MFCC features shape: {padded_X_MFCC.shape}")
    logger.info(f"MFCCT features shape: {padded_X_MFCCT.shape}")
    logger.info(f"Labels shape: {np.array(Y).shape}")
    logger.info("Feature extraction completed successfully.")
    logger.info("Assembling dataset features and labels into a dictionary.")
    dataset_features_and_labels = {
        "data": {
            "mfcc": padded_X_MFCC,
            "mfcct": padded_X_MFCCT
        },
        "labels": np.array(Y, dtype=np.int64),
    }
    
    return dataset_features_and_labels