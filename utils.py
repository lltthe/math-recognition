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

import numpy as np
import cv2
import _pickle as pickle
from keras.models import load_model
import time

WHITE = (255, 255, 255)

def union_rect(a, b):
	x = min(a[0], b[0])
	y = min(a[1], b[1])
	w = max(a[0] + a[2], b[0] + b[2]) - x
	h = max(a[1] + a[3], b[1] + b[3]) - y
	return (x, y, w, h)

def preproc(img):
	t = time.clock()
	
	(H, W) = img.shape[:2]
	image_size = H * W
	
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	
	mser = cv2.MSER_create()
	mser.setMaxArea(image_size // 2)
	mser.setMinArea(1)
	_, rects = mser.detectRegions(img)
	
	
	if len(rects) > 1:
		rects = np.asarray(sorted(rects, key=lambda k : k[0]))
		tmp = []
		pivot = rects[0]
		for r in rects[1:]:
			plower = pivot[0]
			pupper = pivot[0] + pivot[2]
			lower = r[0]
			upper = r[0] + r[2]
			if (lower >= plower and lower <= pupper) or (upper >= plower and upper <= pupper):
				pivot = union_rect(pivot, r)
			else:
				tmp.append(pivot)
				pivot = r
		tmp.append(pivot)
		rects = np.asarray(tmp)
		
	i = 0
	e = 2
	res = []
	for i in range(len(rects)):
		(x, y, w, h) = rects[i]
		x -= e
		y -= e
		w += e
		h += e
		if x < 0:
			x = 0
		if y < 0:
			y = 0
		if x + w > W:
			w = W - x
		if y + h > H:
			h = H - y
		rects[i] = (x, y, w, h)
		
		tmp = img[y:y+h, x:x+w]
		
		if w != h:
			p = (h - w) // 2
			if h < w:
				p = -p
				tmp = cv2.copyMakeBorder(tmp, p, p, 0, 0, cv2.BORDER_CONSTANT, value=WHITE)
			else:
				tmp = cv2.copyMakeBorder(tmp, 0, 0, p, p, cv2.BORDER_CONSTANT, value=WHITE)
		
		tmp = cv2.resize(tmp, (45, 45))
		res.append(tmp.flatten())
		
	return res, rects, (time.clock() - t) * 1000

def load_svm(path):
	with open(path, 'rb') as f:
		svm_model = pickle.load(f)
	return svm_model

def load_cnn(path):
	return load_model(path)

def extra_preproc_cnn(preproced_inp):
	t = time.clock()
	
	res = np.asarray(preproced_inp)
	res = res.astype('float32')
	res /= 255
	res = np.expand_dims(res, axis=2)
	
	return res, (time.clock() - t) * 1000
