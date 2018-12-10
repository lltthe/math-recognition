# -*- coding: utf-8 -*-
"""
Huynh Vinh Loc
Lieng The Phy
Lam Le Thanh The

15CTT - APCS - HCMUS

CS414 - Machine Learning
Final Project - Handwriting Recognition for Basic Arithmetic Expressions

Dec 2018
"""

# MODULES & ENVIRONMENT PREPARATION

import numpy as np
import cv2
from utils import *
import time
import warnings
import sys

warnings.filterwarnings('ignore')

#===================================================
# GLOBAL VARS AND CONSTANTS

brush_thickness = 2
svm_path = 'svm_model4.mdl'
cnn_path = 'cnn_model.h5'

classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'add', 11: 'div', 12: 'minus', 13: 'times'}
 
WD = 'OpenCV - Free Drawing Canvas'

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)

drawing = False
ix = 0
iy = 0

svm_model = load_svm(svm_path)
cnn_model = load_cnn(cnn_path)

#===================================================
# FUNCTIONS

def init_canvas():
	return np.full((180, 1260, 3), WHITE, dtype=np.uint8)

def put(x, y):
	global canvas	
	cv2.line(canvas, (ix, iy), (x, y), BLACK, brush_thickness)
	cv2.imshow(WD, canvas)

def mouse_handle(event, x, y, flags, params):
	global drawing, ix, iy	
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix = x
		iy = y
		put(x, y)
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing:
			put(x, y)
			ix = x
			iy = y
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		put(x, y)
		
def demo(img):
	print('Preprocessing...')
	inps, rects, preproc_time = preproc(img)
	print('\t... in %.3f millisecond(s)\n' % preproc_time)
	
	for r in rects:
		cv2.rectangle(img, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), RED, 1)
	cv2.imshow(WD, img)
	
	print('PREDICTIONS:')
	
	t = time.clock()
	svm_prediction = svm_model.predict(inps)
	t = (time.clock() - t) * 1000
	print('SVM:', list(svm_prediction))
	print('\t... in %.3f millisecond(s)' % t)
	
	inps, cnn_prepare_time = extra_preproc_cnn(inps)
	t = time.clock()
	cnn_prediction = cnn_model.predict_classes(inps)
	t = (time.clock() - t) * 1000
	print('CNN:', [classes[c] for c in cnn_prediction])
	print('\t... in %.3f millisecond(s)' % t)	
	print('\t... with %3f millisecond(s) for extra preparing' % cnn_prepare_time)
	
#===================================================
# MAIN

print('A SIMPLE DEMO FOR BASIC ARITHMETIC MATH EXPRESSION RECOGNITION')
print('\t\tby  Huynh Vinh Loc')
print('\t\t    Lieng The Phy')
print('\t\t    Lam Le Thanh The')
print('\t\t~ 15CTT - APCS - HCMUS ~')
print('\n')

print('* The program can accept one command line argument for an image path...')
print('* Otherwise the program will provide a white canvas for free drawing!\n')

cv2.namedWindow(WD, cv2.WINDOW_AUTOSIZE)

argc = len(sys.argv)

if argc > 1:
	print('* Press the X sign or any key to quit\n')
	try:
		image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
		cv2.imshow(WD, image)
		demo(image)
		while cv2.getWindowProperty(WD, 0) >= 0:
			if (cv2.waitKey(1) != -1):
				break
	except Exception:
		print('ERROR: Cannot open the image, probably due to invalid path, being used, or file corruption!')
else:
	print('* Hold the left mouse button to draw')
	print('* Press ENTER for doing the recognition')
	print('*       SPACEBAR or BACKSPACE to reset the canvas')
	print('*       ESC to quit\n')
	
	canvas = init_canvas()
	cv2.imshow(WD, canvas)
					
	cv2.setMouseCallback(WD, mouse_handle)
			
	j = 11
	while cv2.getWindowProperty(WD, 0) >= 0:
		cmd = cv2.waitKey(1)
		if cmd == 13: # Enter
			old = np.copy(canvas)
			demo(canvas)
			print('\n===================================\n')
			
		elif cmd == ord(' ') or cmd == 8: # Backspace
			canvas = init_canvas()
			cv2.imshow(WD, canvas)
		elif cmd == 27: # Esc
			break
		elif cmd == ord('w'):
			cv2.imwrite('%s.jpg' % j, old)
			j += 1
	
cv2.destroyAllWindows()
#input('FINISHED! PAUSING ON PURPOSE TO SEE RESULTS...\nPress ENTER to quit!')
