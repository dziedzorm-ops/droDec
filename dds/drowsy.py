
from scipy.spatial import distance as dist
from tkinter import *
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


root = Tk()
root.geometry('400x300')
root.configure(bg="yellow")
root.maxsize(400,300)
root.title('DRODSYS')



def sound_alarm():
	playsound.playsound("alert.wav")

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	
	C = dist.euclidean(eye[0], eye[3])

	#eye aspect ratio(EAR)
	ear = (A + B) / (2.0 * C)

	return ear

def start(EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES):


	
	aprs = argparse.ArgumentParser()
	aprs.add_argument("-w", "--webcam", type=int, default=0,
		help="index of webcam on system")
	args = vars(aprs.parse_args())

	
	COUNTER = 0
	ALARM_ON = False

	
	#facial landmark predictor (dlib)
	print("[INFO]facial landmark predictor is loading...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	#video stream thread
	print("[INFO] starting video stream ...")
	vs = VideoStream(src=args["webcam"]).start()
	time.sleep(1.0)

	i = 0
	min_ear = 100
	max_ear = 0
	ear = 0
	
	global text
	while True:
		
		frame = vs.read()
		frame = imutils.resize(frame, width=650)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		cv2.putText(frame, f"EYE_AR_THRESH = {EYE_AR_THRESH}", (10, 480),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(frame, f"EYE_AR_CONSEC_FRAMES = {EYE_AR_CONSEC_FRAMES}", (300, 480),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		# detect faces 
		rects = detector(gray, 0)

		# loop 
		for rect in rects:
			
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			
			ear = (leftEAR + rightEAR) / 2.0

			
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			
			if ear < EYE_AR_THRESH:
				COUNTER += 1

				
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					# turns alarm on if it is off.
					if not ALARM_ON:
						ALARM_ON = True

						
						t = Thread(target=sound_alarm)
						t.deamon = True
						t.start()

					
					cv2.putText(frame, "WAKE UP, DROWSINESS ALERT!", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


			
			else:	
				COUNTER = 0
				ALARM_ON = False

			
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		# show frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		if i<50:
			if ear < min_ear:
				min_ear = ear
			elif ear > max_ear:
				max_ear = ear
		elif i == 50:
			EYE_AR_THRESH = (min_ear + max_ear)/2
	#cleanup
	cv2.destroyAllWindows()
	vs.stop()


#user interface for system using tkinter framework.
def settings():

	settingsBtn.pack_forget()
	startBtn.pack_forget()

	
	EATList = ['0.27','0.28','0.29','0.30','0.31','0.32','0.33','0.34']
	EACList = [44,45,46,47,48,49,50,51,52]
	v1 = StringVar()
	v1.set(0.31)
	v2 = IntVar()
	v2.set(48)

	l1 = Label(root, text="Set Threshold EAR: ", bg="blue", fg="white")
	l1.pack(pady=5)
	opt1 = OptionMenu(root, v1, *EATList)
	opt1.config(width=50, font=('Helvetica', 12))
	opt1.pack()

	l2 = Label(root, text="Set consecutive frames: ", bg="blue", fg="white")
	l2.pack(pady=5)
	opt2 = OptionMenu(root, v2, *EACList)
	opt2.config(width=50, font=('Helvetica', 12))
	opt2.pack()


	ok = Button(root, text="SAVE AND START", command=lambda:start(float(v1.get()), v2.get()), bg="blue", fg="white")
	ok.pack(pady=35)


settingsBtn =  Button(root, text="CONFIGURE", command=settings, bg="blue", fg="white")
settingsBtn.pack(pady=55)

startBtn = Button(root, text="START", command=lambda:start(0.31,48), bg="blue", fg="white")
startBtn.pack(pady=35)

#endBtn = Button(root, text="STOP", command=stop(),bg="black", fg="white")
#endBtn.pack(pady=35)

root.mainloop()
