import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import keyboard
import os

is_pos=False
counter=0

PATH_POS=r"C:\Users\Utente\Documents\development\wakeword_learning\file_audio_pos"
PATH_NEG=r"C:\Users\Utente\Documents\development\wakeword_learning\file_audio_neg"

def record_audio(duration, filename, samplerate=16000):
    print(f"Registrazione per {duration} secondi...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait() 
    wav.write(filename, samplerate, audio) 
    print(f"Audio registrato e salvato come {filename}")

def on_key_press(event):
    global counter
    global is_pos
    if(counter<100):  
        if(event.name=="space"):
            if(is_pos):
                name="pos_sample"+str(counter)+".wav"
                final_name=os.path.join(PATH_POS, name)  
            else:
                name="neg_sample"+str(counter)+".wav"
                final_name=os.path.join(PATH_NEG, name)  

            record_audio(3, final_name, samplerate=16000)
            counter=counter+1
    else:
        exit()



keyboard.on_press(on_key_press)

keyboard.wait()

