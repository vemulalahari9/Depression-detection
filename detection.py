import time
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import numpy as np
from collections import defaultdict
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys
from tkinter import ttk
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr
from gtts import gTTS
import os
import assemblyai as aai
from keras.models import model_from_json
from keras.preprocessing import image

aai.settings.api_key = f"8179381008804b77ad3712a93edc0892"


# nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
recognizer = sr.Recognizer()


main = tkinter.Tk()
main.title("Detecting depression from video and audio")
main.geometry("1400x700")

global filename

detection_model_path = 'models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)

emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprise","neutral"]

def upload():
    global filename
    global value
    filename = askopenfilename(initialdir = "images")
    pathlabel.config(text=filename)
        
def detectExpression():
    global filename

    text.delete("1.0",END)
    c_img = cv2.imread(filename)
    gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_detection.detectMultiScale(gray_img, 1.32, 5)
    text.insert(END,"Total number of faces detected : "+str(len(faces_detected))+"\n\n")
    if len(faces_detected) > 0:
        for (x,y,w,h) in faces_detected:  
            cv2.rectangle(c_img,(x,y),(x+w,y+h),(255,0,0),thickness=2)  
            roi_gray=gray_img[y:y+w,x:x+h] 
            roi_gray=cv2.resize(roi_gray,(48,48))  
            img = roi_gray.reshape((1,48,48,1))
            img = img /255.0

            max_index = np.argmax(emotion_classifier.predict(img.reshape((1,48,48,1))), axis=-1)[0]
                    
            predicted_emotion = EMOTIONS[max_index]
            cv2.putText(c_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            resized_img = cv2.resize(c_img, (1000, 700)) 
            img = cv2.imread('Emoji/'+predicted_emotion+".png")
            img = cv2.resize(img, (600,400))
            cv2.putText(img, "Facial Expression Detected As : "+predicted_emotion, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            text.insert(END,"Facial Expression Detected as : "+predicted_emotion+"\n")
            cv2.imshow('Facial emotion analysis ',resized_img)
            cv2.waitKey(0)
            cv2.imshow('Facial emotion analysis ', img)
            cv2.waitKey(0)
    else:
       messagebox.showinfo("Facial Expression Prediction Screen","No face detceted in uploaded image")


def detect_depression_from_audio():
    text.delete('1.0',END)
    audio_file = filedialog.askopenfilename(initialdir = "audio_files",filetypes=[("Audio Files", "*.wav")])
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        recognized_text = recognizer.recognize_google(audio_data)
        text.insert(END,"\n"+"Recognized text : "+recognized_text+"\n\n")

        tts = gTTS(recognized_text)
        tts.save("output.mp3")
        os.system("start output.mp3")

    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError as e:
        print(f"Error with the API request: {e}")
    else:
        text_scoring(recognized_text)

def text_scoring(recognized_text):
    sentiment_score = sia.polarity_scores(recognized_text)['compound']
    if sentiment_score <= -0.5:
        text.insert(END,"Detected as : High risk of depression"+"\n")
    elif -0.5 < sentiment_score < 0.5:
        text.insert(END,"Detected as : Mild risk of depression"+"\n")
    else:
        text.insert(END,"Detected as : No depression detected"+"\n")


def detectfromvideo(image):
    result = 'none'
    temp = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    output = "none"
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = temp[fY:fY + fH, fX:fX + fW]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        output = label
    else:
        print('no faces are founded')   
    return output



def detect_depression_from_audio1():
    text.delete('1.0',END)
    text.insert(END,"Dont press any key it is under processing..........."+"\n\n\n")
    audio_file = filedialog.askopenfilename(initialdir = "",filetypes=[("Audio Files", "*.wav")])
    config = aai.TranscriptionConfig(speaker_labels=True)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
    audio_file,
    config=config
    )

    for utterance in transcript.utterances:
        if utterance.speaker == "C":
            text.insert(END,f"Speaker {utterance.speaker}: {utterance.text}"+"\n")
            text_scoring(utterance.text)
        else:
            text.insert(END,f"Speaker {utterance.speaker}: {utterance.text}"+"\n")


def detectWebcamExpression():
    text.delete('1.0',END)
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        height, width, channels = img.shape

        result = detectfromvideo(img)
        if result != 'none':
            img1 = cv2.imread('Emoji/'+result+".png")
            img1 = cv2.resize(img1, (width,height))
            cv2.putText(img1, "Facial Expression Detected As : "+result, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
            cv2.imshow("Emoji Output",img1)
        cv2.putText(img, "Facial Expression Detected As : "+result, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        cv2.imshow("Facial Expression Output", img)
        if cv2.waitKey(650) & 0xFF == ord('q'):
            break   
    cap.release()
    cv2.destroyAllWindows()


font = ('times', 22, 'bold')
title = Label(main, text='Detecting Depression From Video and Audio')
title.config(bg='light pink', fg='blue')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Image With Face", command=upload)
upload.place(x=50,y=150)
upload.config(font=font1,bg="black",fg='white')  

pathlabel = Label(main)
pathlabel.config(bg='violet', fg='black')  
pathlabel.config(font=font1)           
pathlabel.place(x=300,y=150)

emotion = Button(main, text="Detect Facial Expression", command=detectExpression)
emotion.place(x=50,y=230)
emotion.config(font=font1,bg="black",fg='white')

emotion = Button(main, text="Detect Facial Expression from WebCam", command=detectWebcamExpression)
emotion.place(x=300,y=230)
emotion.config(font=font1,bg="black",fg='white') 

audio = Button(main, text="Detect Depression Through single Voice", command=detect_depression_from_audio)
audio.place(x=670,y=230)
audio.config(font=font1,bg="black",fg='white') 

audio1 = Button(main, text="Detect Depression Through multi Voices", command=detect_depression_from_audio1)
audio1.place(x=1035,y=230)
audio1.config(font=font1,bg="black",fg='white') 

font1 = ('times', 14, 'bold')
text=Text(main,height=15,width=137)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=310)
text.config(font=font1)


main.config(bg='light pink')
main.mainloop()
