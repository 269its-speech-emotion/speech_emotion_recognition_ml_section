EMOTION_LABELS = (
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)

EMOTION_TO_INDEX = {
    emotion: index 
    for index, emotion in enumerate(EMOTION_LABELS)
}

INDEX_TO_EMOTION = {
    index: emotion 
    for emotion, index in EMOTION_TO_INDEX.items()
}

RAVDESS_EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

SAMPLE_RATE = 16000  # Sample rate for audio files
FRAME_LENGTH_MS = 25 # Frame length in milliseconds
FRAME_STEP_MS = 10  # Frame step in milliseconds
N_MFCC = 13 # Number of MFCC coefficients
N_MELS = 16 # Number of Mel filter banks
MFCCT_BIN_SIZE = 1500  # Size of each MFCC bin