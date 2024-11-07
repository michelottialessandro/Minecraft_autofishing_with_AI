import sounddevice as sd
import numpy as np
import librosa
import joblib
import pyautogui
import time
# Simula il clic del tasto destro del mouse
pyautogui.rightClick()

def record_audio(duration, samplerate=16000):
    print(f"Registrazione per {duration} secondi...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  
    return audio.flatten()

def predict_live_sound(audio, samplerate, model,is_svm):
    mfccs = librosa.feature.mfcc(y=audio.astype(float), sr=samplerate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0).reshape(1, -1)  
    prediction = model.predict(mfccs)
    if(is_svm):
        if prediction[0] == 0:
            print("Suono positivo riconosciuto!")
            return True
        else:
            print("Suono negativo riconosciuto.")
            return False
    else:
        if prediction[0] == 1:
            print("Suono positivo riconosciuto!")
            return True
        else:
            print("Suono negativo riconosciuto.")
            return False
        
clf = joblib.load('modello_minecraft_clf.pkl')

#You might want to decrease the duration = 0.75 parameter in the auto_fishing.py file to increase the reactivity in fishing but be careful, an excessive decrease of this parameter causes a loss in the accuracy of the model.
duration = 0.75  #not more than 3s not less then 0.50
samplerate = 16000  
while True:
    audio = record_audio(duration, samplerate)
    if(predict_live_sound(audio, samplerate, clf,is_svm=False)):
        pyautogui.rightClick()
        time.sleep(0.5)
        pyautogui.rightClick()

        

