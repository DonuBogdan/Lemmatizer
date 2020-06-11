from tkinter import *
from scipy.io.wavfile import write

import sounddevice as sd
import numpy as np
import speech_recognition as sr
from tkinter import messagebox 


# Variables
fs = 44100  # Sample rate
seconds = 5  # Duration of recording

root = Tk(className = 'Lemmatizer')

# Set window size
root.geometry('700x400')

# Set window color
root['background'] = '#CDE1F5'

def get_lemma():

    messagebox.showinfo(title = 'Info 1', message = 'The lemmatization process begun !')
    button_2.invoke()

    # myrecording = sd.rec(int(seconds * fs), samplerate = fs, channels = 2)
    # sd.wait()  # Wait until recording is finished

    # # https://stackoverflow.com/questions/52249985/python-speech-recognition-tool-does-not-recognize-wav-file
    # y = (np.iinfo(np.int32).max * (myrecording/np.abs(myrecording).max())).astype(np.int32)

    # write('output.wav', fs, y)  # Save as WAV file 

    # print(type(myrecording))

    r = sr.Recognizer()
    demo = sr.AudioFile('demo_ro.wav')

    print(demo)

    with demo as source:
        audio = r.record(source)

    print(audio)

    print(type(audio))

    text = r.recognize_google(audio, language = 'ro-RO')
    print(text)
    print(type(text))
    print(text.split()[-1])
    
    lemma  = text.split()[-1]
    
    print(lemma)

    # Display label_1 after getting the lemma
    label_1.configure(text = 'Lemma: ' + lemma)
    label_1.place(relx = 0.5, rely = 0.60, anchor = CENTER)

    print('Done')

    messagebox.showinfo(title = 'Info 2', message = 'The lemmatization process is over !')
    

record_clear_frame = Frame(root)
record_clear_frame.place(relx = 0.5, rely = 0.45, anchor = CENTER)

# Define widgets
label_0 = Label(root, text = 'Lemmatizer', bg = '#CDE1F5', font = ('Helvetica', 20, 'italic'))
label_1 = Label(root, bg = '#CDE1F5', font = ('Helvetica', 14))
button_1 = Button(record_clear_frame, text = 'Record', command = lambda: [label_1.place_forget(), get_lemma()], font = ('Helvetica', 14), bg = '#3385ff', activebackground = '#3385ff')
button_2 = Button(record_clear_frame, text = 'Clear', command = lambda: label_1.place_forget(), font = ('Helvetica', 14), bg = '#008000', activebackground = '#008000')

label_0.pack()
button_1.pack(side = 'left')
button_2.pack(side = 'right')

root.mainloop()







