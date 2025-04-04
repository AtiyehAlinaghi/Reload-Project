######################################################################################################
#
# This script is designed for use with the 'Reload' project dataset.
#
# Make sure the following are placed in your base folder:
# - The 'reload' folder (containing participant audio recordings)
# - An Excel sheet containing the data summary
#
# The script performs the following steps:
# 1. Reads the Excel sheet to retrieve participant and recording metadata
# 2. Iterates through the folders to read and process the audio files
#
# To apply an HMM (Hidden Markov Model) for tracking illness progression, 
# we recommend selecting participants with at least 3 days of recordings.
#
# To minimize the effect of underlying conditions, we also select on participants 
# with similar demographic profiles.
#
# For feature extraction, we use different configurations of the openSMILE library 
# to explore various subgroups of acoustic features.
#
# A group of features have been selected based on this paper:
# Xu, Chenzi, et al.(2023) 
# "Contributions of acoustic measures to the classification of laryngeal voice quality in continuous
# English speech." Proceedings of the International Congress of Phonetic Sciences (ICPhS). 
# 
#####################################################################################################
#
# Last modified by Atiyeh Alinaghi 02/04/2025 03:16 p.m.
#
# University of Southampton, ISVR
#
######################################################################################################


import os
import pandas as pd
import opensmile
import soundfile as sf  # For reading .wav files
from pydub import AudioSegment  # For reading .m4a files
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import numpy as np
import librosa
import json
# import librosa 

# Load JSON data
# This is just a random file to check if json.load() works
with open('daily.json', 'r') as file:
    data = json.load(file)

# Select the set of features & the related configuration
# Harmonics & Formants, MFCC or LLD
Feature_set = 'Harmonics & Formants'

if Feature_set == 'Harmonics & Formants':
    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b, # To include harmonic & formant features
    )   
elif Feature_set == 'MFCC':
    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase, # To include MFCC features
    )
elif Feature_set == 'LLD':
    smile = opensmile.Smile(
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
else:
    smile = opensmile.Smile(
    feature_level=opensmile.FeatureLevel.Functionals,
    )



def load_participant_ids_from_excel(excel_file, column_name):
    """
    Load participant IDs from an Excel file.
    
    Args:
        excel_file (str): Path to the Excel file.
        column_name (str): Name of the column containing participant IDs.
    
    Returns:
        list: A list of participant IDs.
    """
    df = pd.read_excel(excel_file)
    return df[column_name].dropna().tolist()


def read_audio_file(file_path):
    """
    Reads an audio file (.wav or .m4a) and returns its data and sampling rate.
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    if file_path.endswith(".wav"):
        audio_data, samplerate = sf.read(file_path) 
        print('wav file')         
    elif file_path.endswith(".m4a"):
        audio_data = AudioSegment.from_file(file_path) 
        print('m4a file')
    else:
        raise ValueError(f"Unsupported file type for {file_path}")
    
    return audio_data


def find_and_read_audio_files(base_folder, participant_ids, feature_group, audio_type):
    """
    Traverse the folder structure, find vowel .wav or .m4a files, and store their data.
    
    Args:
        base_folder (str): Path to the base folder.
        participant_ids (list): List of participant IDs (folder names).
        feature_group (str): Defines a set of features to be selected.
        audio_type (str) for m4a files: {'breathe', 'cough', 'exhale', 'read', 'vowel'}
        audio_type (str) for wav files: {'breath', 'cough', 'exhalation', 'text', 'vowel'}

    Returns:
        
    """
    # 
    if feature_group == 'formants_mean':
       # The related features based on the Xu, Chenzi, et al.(2023) 
        selected_features_names = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'jitterLocal_sma3nz_amean',
        'shimmerLocaldB_sma3nz_amean', 'HNRdBACF_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_amean',
        'F1frequency_sma3nz_amean', 'F1bandwidth_sma3nz_amean', 'F1amplitudeLogRelF0_sma3nz_amean', 'F2frequency_sma3nz_amean',
        'F2amplitudeLogRelF0_sma3nz_amean', 'loudnessPeaksPerSec']
    elif feature_group == 'formants_std':
         # The related features based on the Xu, Chenzi, et al.(2023) 
        selected_features_names = ['F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'loudness_sma3_stddevNorm', 'jitterLocal_sma3nz_amean',
        'shimmerLocaldB_sma3nz_amean', 'HNRdBACF_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_amean',
        'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_stddevNorm',
        'F2amplitudeLogRelF0_sma3nz_stddevNorm', 'F3frequency_sma3nz_stddevNorm','F3amplitudeLogRelF0_sma3nz_stddevNorm']
    elif feature_group == 'mfcc':
        selected_features_names = ['mfcc_sma[1]_max', 'mfcc_sma[1]_min', 'mfcc_sma[1]_amean', 'mfcc_sma[2]_max', 'mfcc_sma[2]_min', 'mfcc_sma[2]_amean',
        'mfcc_sma[3]_max', 'mfcc_sma[3]_min', 'mfcc_sma[3]_amean', 'mfcc_sma[4]_max', 'mfcc_sma[4]_min', 'mfcc_sma[4]_amean',
        'mfcc_sma[5]_max', 'mfcc_sma[5]_min', 'mfcc_sma[5]_amean']
    else:
        selected_features_names = []
    
    # The audio file names are different for different audio formats
    if audio_type in {"breath","exhalation","text"}:
        audio_type_wav = audio_type
    elif audio_type in {"breathe", "exhale", "read"}:
        audio_type_m4a = audio_type
    elif audio_type in {"cough","vowel"}:
        audio_type_wav = audio_type
        audio_type_m4a = audio_type
    else:
        raise KeyError("The provided audio_type is not recognized.")

    num_features = len(selected_features_names) # Just for now
    vowel_data = {}
    audio_read = False
    # Initialize an empty DataFrame to hold all features
    full_features = pd.DataFrame()
    full_labels = pd.DataFrame()
    
    stack_features = np.zeros((0,num_features), dtype='float')
    Lengths =[]
    p = 0 # number of valid participants
    for participant_id in participant_ids:
        participant_folder = os.path.join(base_folder, str(participant_id))
        if not os.path.exists(participant_folder):
            print(f"Folder {participant_folder} does not exist. Skipping.")
            continue

        # Initialize dictionary list to store vowel files for this participant
        vowel_data[participant_id] = []
        
        each_ID_days_features = np.zeros((0,num_features), dtype='float')
        num_days = 0
        p = p+1
        # participant_folder is like ~/reload/7gz4amS9x3
        for subfolder in os.listdir(participant_folder):
            # subfolder goes through all the dates for each participants & 
            # is like 2024-11-14-14_48_26_431590
            subfolder_path = os.path.join(participant_folder, subfolder)
            # subfolder_path is like ~/reload/7gz4amS9x3/2024-11-14-14_48_26_431590
            if os.path.isdir(subfolder_path):  # Only process directories
                # Look for files with "vowel" in their name and supported extensions
                for file in os.listdir(subfolder_path):
                    # if "vowel" in file.lower() and file.endswith(".wav"):
                    # if "cough" in file.lower() and file.endswith(".wav"): 
                    if audio_type_wav in file.lower() and file.endswith(".wav"): 
                        file_path = os.path.join(subfolder_path, file)
                        audio_data = read_audio_file(file_path)
                        # Load audio file
                        audio_file, sr = librosa.load(file_path, sr=None)
                        # Normalize audio
                        normalized_audio = librosa.util.normalize(audio_file)
                        # Save normalized audio
                        sf.write('normalized_audio.wav', normalized_audio, sr)
                        # features = smile.process_file(file_path)
                        features = smile.process_file('normalized_audio.wav')
                        os.remove('normalized_audio.wav')
                        # full_features = pd.concat([full_features, features], ignore_index=True)
                        print("claculating the features")
                        audio_read = True
                        # Store as tuple (file_path, audio_data)
                        vowel_data[participant_id].append((audio_data))                            
                    # elif "vowel" in file.lower() and file.endswith(".m4a"):
                    # elif "cough" in file.lower() and file.endswith(".m4a"):
                    elif audio_type_m4a in file.lower() and file.endswith(".m4a"):
                        file_path = os.path.join(subfolder_path, file)
                        audio_data_object = read_audio_file(file_path)
                        audio_data = audio_data_object.get_array_of_samples()
                        # Load audio file
                        audio_file, sr = librosa.load(file_path, sr=None)
                        # Normalize audio
                        normalized_audio = librosa.util.normalize(audio_file)
                        # Include extra code or a function to divide audio files—especially cough
                        # recordings—into smaller chunks for better analysis. 
                        # Save normalized audio
                        sf.write('normalized_audio.wav', normalized_audio, sr)
                        # features = smile.process_file(file_path)
                        features = smile.process_file('normalized_audio.wav')
                        os.remove('normalized_audio.wav')
                        # full_features = pd.concat([full_features, features], ignore_index=True)
                        audio_read = True
                        # Store as tuple (file_path, audio_data)
                        # vowel_data[participant_id].append((file_path, audio_data))
                        vowel_data[participant_id].append((audio_data))
                    if "daily" in file.lower() and file.endswith(".json"): # and audio_read:
                        # Load JSON data
                        # Get the current working directory
                        current_directory = os.getcwd()
                        # Join the directory with the filename
                        file_path = os.path.join(current_directory, subfolder_path, file)
                        with open(file_path, 'r') as json_file:
                            daily_data = json.load(json_file)
                        if daily_data.get("currently_with_rti") == "hasRTI":
                            print("The value of 'currently_with_rti' is 'hasRTI'.")
                            label = pd.DataFrame([1])
                        else:
                            print("The value of 'currently_with_rti' is 'noRTI'.")
                            label = pd.DataFrame([0])
                        # audio_read = False
                        json_read = True
                if audio_read and json_read:
                    selected_features = features[selected_features_names]
                    # full_features = pd.concat([full_features, features], ignore_index=True)
                    # full list of selected features for all participants
                    full_features = pd.concat([full_features, selected_features], ignore_index=True)   
                    full_labels = pd.concat([full_labels, label], ignore_index=True)
                    audio_read = False
                    json_read = False 
                    day_feature = selected_features.to_numpy()
                    # day-by-day features for each participant
                    each_ID_days_features = np.vstack([each_ID_days_features,day_feature])
                    num_days = num_days+1
        stack_features = np.vstack([stack_features,each_ID_days_features])
        Lengths.append(num_days)
        print('another participant:', p)

    return vowel_data, full_features, full_labels, stack_features, Lengths



# Example Usage

# A subset of reload data up to 18/11/2024 with at least 3 days of recordings 
# excel_file = "Day3_reload_18-11-2024.xlsx" 

# An example of the data summary updated on 31/12/2024. You can get a new update from reload 
# Run the Matlab file, 'Import_reload_data_summary.m', provided to import 
# the summary of the dataset
# excel_file = "Data_Reload_03032025_daily_update.xlsx"  
excel_file = "Reload_Data_same_demog.xlsx"  # Replace with the path to your Excel file
column_name = "Participant ID"  # Replace with the actual column name in your Excel file
base_folder = "reload"  # Replace with the path to your base folder

# Step 1: Load participant IDs from the Excel file
participant_ids = load_participant_ids_from_excel(excel_file, column_name)
# Step 2: Find and read vowel /a/ .wav files
feature_group = 'formants_mean' # 'formants_std' or 'mfcc'
audio_type = "cough"
vowel_data, full_features, full_labels, stack_features, Lengths = find_and_read_audio_files(base_folder, participant_ids,feature_group, audio_type)


selected_features = full_features
# selected_features.to_csv("RTI_vowel_samedemog_Xu_features.csv", index=False)
# selected_features.to_csv("RTI_vowel_samedemog_mfcc_features.csv", index=False)
# selected_features.to_csv("RTI_vowel_samedemog_std_features.csv", index=False)
# selected_features.to_csv("RTI_cough_samedemog_std_features.csv", index=False)
# selected_features.to_csv("RTI_cough_samedemog_Xu_mean_features.csv", index=False)
selected_features.to_csv("RTI_cough_normalized_samedemog_Xu_std_features.csv", index=False)
# full_features.to_csv("RTI_vowel_features.csv", index=False)
# full_labels.to_csv("RTI_vowel_labels.csv", index=False)
# full_labels.to_csv("RTI_vowel_samedemog_labels.csv", index=False)
full_labels.to_csv("RTI_cough_samedemog_labels.csv", index=False)
# Convert dictionary to DataFrame and write directly to CSV
# df = pd.DataFrame(vowel_data)
# df.to_csv("RTI_vowel_data.csv", index=False)

num_samples = stack_features.shape[0]
train_len = int(0.8*num_samples)
test_len = num_samples-train_len # int(0.2*num_samples)
features_train = stack_features[0:train_len,:]
cumsum = np.cumsum(Lengths)
# Find the first index where the cumulative sum is greater than or equal to the target sum
num_train = np.where(cumsum == train_len)
lengths_train = Lengths[0:num_train[0][0]+1]
features_test = stack_features[train_len:,:]
# Initialize and train the HMM model
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
model.fit(features_train, lengths=lengths_train)
log_likelihood = model.score(features_test)


# Checking the model parameters
# print("Start probabilities:", model.startprob_)
# print("Transition matrix:", model.transmat_)
print("Means of each state:", model.means_)
print("Covariances of each state:", model.covars_)
print("log_likelihood:", log_likelihood)

