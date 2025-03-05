library(reticulate)

# Define Python script inside R
py_run_string("
import essentia.standard as es
import os
import pandas as pd

AUDIO_FOLDER = 'file:///Users/reinout/Desktop/uva/14_compmus/homework/homework%20week%208/compmus-corpus-2025/'
data = []

def extract_features(file_path):
    loader = es.MonoLoader(filename=file_path)
    audio = loader()
    
    features = {
        'filename': os.path.basename(file_path),
        'zcr': es.ZeroCrossingRate()(audio),
        'rms': es.RMS()(audio),
        'spectral_centroid': es.SpectralCentroidTime()(audio),
        'spectral_flux': es.SpectralFlux()(audio),
        'spectral_rolloff': es.RollOff()(audio),
        'mfcc': es.MFCC()(audio)[1].mean(),
        'bpm': es.BeatTrackerMultiFeature()(audio),
    }
    return features

for file in os.listdir(AUDIO_FOLDER):
    if file.endswith('.wav') or file.endswith('.mp3'):
        file_path = os.path.join(AUDIO_FOLDER, file)
        try:
            features = extract_features(file_path)
            data.append(features)
        except Exception as e:
            print(f'Error processing {file}: {e}')

df = pd.DataFrame(data)
df.to_csv('audio_features.csv', index=False)
")

# Load the extracted CSV into R
df <- read.csv("audio_features.csv")

# Preview the first few rows
head(df)
