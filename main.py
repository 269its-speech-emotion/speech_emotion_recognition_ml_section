################################################################################
#                                                                              #
#                   This is the features preparation part of the project       #
#                                                                              #
################################################################################

from from_root import from_root
from ser.make_dataset import organize_dataset_into_emotion_type, create_production_data
from ser.visualize import get_random_audio_files, plot_audio_features
from ser.extract_features import extract_dataset_features_and_labels
from ser.models.train_model import prepare_and_split_data, model_training
from ser.evaluate_model import model_performance_and_assessment
from ser.constants import EMOTION_LABELS 


RAW_AUDIO_DATASET_FOLDER = from_root('data', 'ravdess_raw')
ORGANIZED_AUDIO_DATASET_FOLDER = from_root('data', 'ravdess_organized')
PRODUCTION_AUDIO_DATASET_FOLDER = from_root('data', 'ravdess_production')
REPORT_FOLDER = from_root('reports')

#LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
FEATURES_TYPES = ['spectrogram', 'mfcc', 'mel_log_spectrogram']

if __name__ == "__main__":
    # First let arrange the folders by speech-emotion type
    print("Let arrange the folders by speech-emotion type")
    #organize_dataset_into_emotion_type(origin_data_folder=RAW_AUDIO_DATASET_FOLDER,
    #                                organized_data_folder=ORGANIZED_AUDIO_DATASET_FOLDER)

    # Prepare the production data
    print("Prepare the production data")
    #create_production_data(processed_data_folder=ORGANIZED_AUDIO_DATASET_FOLDER,
    #                    production_data_folder=PRODUCTION_AUDIO_DATASET_FOLDER,
    #                    n_sample=5)

    # Get random audio files
    print("Get random audio files")
    #random_audio_samples = get_random_audio_files(organized_audio_data_folder=ORGANIZED_AUDIO_DATASET_FOLDER,
    #                                        n_sample=6)
    # Plot the random audio files
    print("Make the plot of the audio characteristics.")
    #plot_audio_features(audio_samples=random_audio_samples, n_samples=6)

    print("Extract features and labels from the organized audio dataset.")
    data_X_Y = extract_dataset_features_and_labels(organized_audio_data_folder=ORGANIZED_AUDIO_DATASET_FOLDER)
    dict_data_X_Y = {'data': data_X_Y[0], 'labels': data_X_Y[1]}

    
    for feature_type in ['mfcc']:#FEATURES_TYPES:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_and_split_data(data=dict_data_X_Y,
                                                                                feature_type=feature_type)
        print(f"Train features: {X_train.shape}, Train labels: {y_train.shape}")
        print(f"Val features: {X_val.shape}, Val labels: {y_val.shape}")
        print(f"Test features: {X_test.shape}, Test labels: {y_test.shape}")

        # Train the model according to the model type
        history, model = model_training(
            x_train=X_train, 
            x_val=X_val, 
            y_train=y_train, 
            y_val=y_val, 
            n_classes=8
        )

        # Call the function to visualise the performance of the model in training and assess the model
        print(f"\nmodel performance in training & Assessment of the model ")
        model_performance_and_assessment(history=history, model=model, x_test=X_test, y_test=y_test, class_names=EMOTION_LABELS,
                                         model_type=feature_type, report_folder=REPORT_FOLDER)
                                         