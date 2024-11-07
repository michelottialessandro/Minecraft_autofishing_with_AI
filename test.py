import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import joblib
from sklearn.svm import SVC

iteration=95

positive_dir = r'C:\Users\Utente\Documents\development\wakeword_learning\file_audio_pos'  
negative_dir = r'C:\Users\Utente\Documents\development\wakeword_learning\file_audio_neg' 

clf = joblib.load('modello_minecraft_clf.pkl')

list_neg_clf=[]

list_pos_clf=[]

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
        return 1
    else:
        print(f"{file_path}: Suono negativo riconosciuto.")
        return 0

for i in range(iteration):
    path="pos_sample"+str(i)+".wav"
    file_path = os.path.join(positive_dir, path)
    list_pos_clf.append(predict_sound(file_path,clf))

for i in range(iteration):
    path="neg_sample"+str(i)+".wav"
    file_path = os.path.join(negative_dir, path)
    list_neg_clf.append(predict_sound(file_path,clf))
    
counter_right_clf0=0
counter_right_clf1=0

for el in list_neg_clf:
    if(el==0):
        counter_right_clf0=counter_right_clf0+1
        

for el in list_pos_clf:
    if(el==1):
        counter_right_clf1=counter_right_clf1+1



print(f"correttezza su dateset positivo clf: {(counter_right_clf1/iteration)*100}")
print(f"correttezza su dateset negativo clf: {(counter_right_clf0/iteration)*100}")
