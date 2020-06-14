from tkinter import *
from scipy.io.wavfile import write

import sounddevice as sd
import numpy as np
import speech_recognition as sr

from tkinter import messagebox 

from predict import predict_func


# Variables
fs = 44100  # Sample rate
seconds = 25  # Duration of recording


def get_lemmas():

    messagebox.showinfo(title = 'Info 1', message = 'The lemmatization process begun !')
    button_2.invoke()

    myrecording = sd.rec(int(seconds * fs), samplerate = fs, channels = 2)
    sd.wait()  # Wait until recording is finished

    # https://stackoverflow.com/questions/52249985/python-speech-recognition-tool-does-not-recognize-wav-file
    y = (np.iinfo(np.int32).max * (myrecording/np.abs(myrecording).max())).astype(np.int32)

    write('output.wav', fs, y)  # Save as WAV file 

    r = sr.Recognizer()
    demo = sr.AudioFile('output.wav')

    with demo as source:
        audio = r.record(source)

    text = r.recognize_google(audio, language = 'ro-RO')
    
    word_1 = text.split()[-3]
    word_2  = text.split()[-1]
  
    list_of_predictions = predict_func([word_1, word_2])

    # Display label_1 after getting the lemma
    label_1.configure(text = 'Lemma: ' + list_of_predictions[0])
    label_1.place(relx = 0.5, rely = 0.60, anchor = CENTER)

    # Display label_2 after getting the lemma
    label_2.configure(text = 'Lemma: ' + list_of_predictions[1])
    label_2.place(relx = 0.5, rely = 0.70, anchor = CENTER)

    messagebox.showinfo(title = 'Info 2', message = 'The lemmatization process is over !')
    

root = Tk(className = 'Lemmatizer')
# Set window size
root.geometry('700x400')
# Set window color
root['background'] = '#CDE1F5'   

record_clear_frame = Frame(root)
record_clear_frame.place(relx = 0.5, rely = 0.45, anchor = CENTER)

# Define widgets
label_0 = Label(root, text = 'Lemmatizer', bg = '#CDE1F5', font = ('Helvetica', 20, 'italic'))
label_1 = Label(root, bg = '#CDE1F5', font = ('Helvetica', 14))
label_2 = Label(root, bg = '#CDE1F5', font = ('Helvetica', 14))
button_1 = Button(record_clear_frame, text = 'Record', command = lambda: [label_1.place_forget(), label_2.place_forget(), get_lemmas()], font = ('Helvetica', 14), bg = '#3385ff', activebackground = '#3385ff')
button_2 = Button(record_clear_frame, text = 'Clear', command = lambda: [label_1.place_forget(), label_2.place_forget()], font = ('Helvetica', 14), bg = '#008000', activebackground = '#008000')

label_0.pack()
button_1.pack(side = 'left')
button_2.pack(side = 'right')

root.mainloop()