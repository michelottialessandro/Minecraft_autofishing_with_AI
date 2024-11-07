import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import joblib
from sklearn.svm import SVC

# Caricamento dell'audio e estrazione di caratteristiche (MFCC)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_sound(file_path, model):
    mfccs = extract_features(file_path)
    mfccs = mfccs.reshape(1, -1)  # Rimodella l'input per il modello

    prediction = model.predict(mfccs)

    if prediction[0] == 1:
        print(f"{file_path}: Suono positivo riconosciuto!")
    else:
        print(f"{file_path}: Suono negativo riconosciuto.")


def load_data(positive_dir, negative_dir):
    features = []
    labels = []

    for file_name in os.listdir(positive_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(positive_dir, file_name)
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(1)  
            
    for file_name in os.listdir(negative_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(negative_dir, file_name)
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(0)  
    return np.array(features), np.array(labels)

positive_dir = r'C:\Users\Utente\Documents\development\wakeword_learning\file_audio_pos'  
negative_dir = r'C:\Users\Utente\Documents\development\wakeword_learning\file_audio_neg' 

# load data and labels
features, labels = load_data(positive_dir, negative_dir)
#split data in training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

#training  2 models one is randon forest classifier the other supported vector machine
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
accuracy_svm=svm.score(X_test,y_test)
print(f'Accuratezza clf: {accuracy * 100:.2f}%')
print(f'Accuratezza svm: {accuracy_svm * 100:.2f}%')

predict_sound("audio_neg.wav",clf)
predict_sound("audio_neg.wav",svm)

#saving the 2 models
joblib.dump(clf, 'modello_minecraft_clf.pkl')
joblib.dump(svm, 'modello_minecraft_svm.pkl')

print("Modello salvato come modello_minecraft_clf.pkl")
print("Modello salvato come modello_minecraft_svm.pkl")