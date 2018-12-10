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
import os
from os.path import join
from utils import *
from time import clock
import warnings
import csv

warnings.filterwarnings('ignore')

result_path = 'test_result.csv'
svm_path = 'svm_model4.mdl'
cnn_path = 'cnn_model.h5'

folder_path = 'test'
files = os.listdir(folder_path)

imgs = []
labels = []

try:	
	for f in files:
		imgs.append(cv2.imread(join(folder_path, f), cv2.IMREAD_COLOR))
	
		l = f.lower()
		l = l.split('.')[0]
		l = l.split(' ')
		labels.append(l)
		
except Exception:
	print('ERROR: Loading test images failed!')
	
svm_model = load_svm(svm_path)
cnn_model = load_cnn(cnn_path)

classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'add', 11: 'div', 12: 'minus', 13: 'times'}
	
def match(prediction, label):
	n = len(prediction)
	if n != len(label):
		return 0
	
	s = 0
	for i in range(n):
		if prediction[i] == label[i]:
			s += 1
	return s

def proc(img, label):
	inps, _, preproc_time = preproc(img)
	
	t = clock()
	svm_prediction = svm_model.predict(inps)
	svm_time = (time.clock() - t) * 1000
	svm_prediction = list(svm_prediction)
	svm_count_matches = match(svm_prediction, label)
	
	inps, cnn_prepare_time = extra_preproc_cnn(inps)
	t = clock()
	cnn_prediction = cnn_model.predict_classes(inps)
	cnn_prediction = [classes[c] for c in cnn_prediction]
	cnn_time = (time.clock() - t) * 1000
	cnn_count_matches = match(cnn_prediction, label)
	
	return svm_prediction, cnn_prediction, svm_count_matches, cnn_count_matches, preproc_time, svm_time, cnn_time, cnn_prepare_time

def list2str(l):
	return ' '.join(l)

with open(result_path, 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(['No.', 'Label', 'SVM Prediction', 'CNN Prediction', 'Size', 'SVM Matches', 'SVM Accuracy', 'CNN Matches', 'CNN Accuracy', 'Shared Preprocessing Time', 'SVM Running Time', 'CNN Running Time', 'CNN Extra Preparation Time'])
	
	for i in range(len(imgs)):
		img = imgs[i]
		label = labels[i]
		svm_prediction, cnn_prediction, svm_count_matches, cnn_count_matches, preproc_time, svm_time, cnn_time, cnn_prepare_time = proc(img, label)
		
		n = len(label)
		writer.writerow([i+1, list2str(label), list2str(svm_prediction), list2str(cnn_prediction), n, svm_count_matches, svm_count_matches / n, cnn_count_matches, cnn_count_matches / n, preproc_time, svm_time, cnn_time, cnn_prepare_time])
