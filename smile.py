import cv2
import numpy as np
from time import sleep
from datetime import datetime

def display_text(image, txt, nivel, color):

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (25,40*nivel)
	fontScale              = 1
	#fontColor              = (255,255,255)
	lineType               = 2

	cv2.putText(image,str(txt), 
	    bottomLeftCornerOfText, 
	    font, 
	    fontScale,
	    color,
	    lineType)

	return image


def hora_atual():

	now = datetime.now()

	d = str(now.day)
	if len(d) < 2:
		d = '0'+d

	mon = str(now.month)
	if len(mon) < 2:
		mon = '0'+mon

	y = str(now.year)
	if len(y) < 2:
		y = '0'+y

	h = str(now.hour)
	if len(h) < 2:
		h = '0'+h

	m = str(now.minute)
	if len(m) < 2:
		m = '0'+m

	s = str(now.second)
	if len(s) < 2:
		s = '0'+s

	return d+'-'+mon+'-'+y+'-'+h+'-'+m+'-'+s


captura = cv2.VideoCapture(0)

segs = 0
c_smiles = 0
i = 0

while(True):
	ret, frame = captura.read()

	save_image = frame.copy()

	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	image = cv2.equalizeHist(image)

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	faces  = face_cascade.detectMultiScale(image, scaleFactor = 1.3, minNeighbors = 5)

	for (sx, sy, sw, sh) in faces:
		cv2.rectangle(image, (sx, sy), ((sx + sw), (sy + sh)), (255, 0,0), 2)
		roi = image[sy:sy+sh,sx:sx+sw]

		roi = cv2.equalizeHist(roi)

		image[sy:sy+sh,sx:sx+sw] = roi
		
		smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
		smiles  = smile_cascade.detectMultiScale(roi, scaleFactor = 1.8, minNeighbors = 30)

		c_smiles = len(smiles)

		for (sxs, sys, sws, shs) in smiles:
			mx = int(sxs+(sws/2))+sx
			my = int(sys+(shs/2))+sy

			sxs = sxs + sx
			sys = sys + sy

			cv2.rectangle(image, (mx, my), ((mx + 2), (my + 2)), (0, 0, 255), 2)
			cv2.rectangle(image, (sxs, sys), ((sxs + sws), (sys + shs)), (0, 255,0), 2)

	i = i+1

	if i%2 == 0:
		if c_smiles > 0:
			if segs < 5:
				segs = segs + 1
				#print('Segundos: '+str(segs))
			else:
				segs = 0
				#print('Take')
				frame = display_text(frame, 'TIREI !', 4, (0,255,0))
				cv2.imwrite(hora_atual()+'.jpg', save_image)
		else:
			segs = 0

	frame = display_text(frame, 'Contagem: '+str(segs), 1, (255,255,255))
	frame = display_text(frame, 'Sorrisos: '+str(c_smiles), 2, (255,255,255))

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	if k == 116:
		frame = display_text(frame, 'TIREI !', 4, (0,255,0))
		cv2.imwrite(hora_atual()+'.jpg', save_image)

	cv2.imshow("Smile Detector", frame)