"""
Huynh Vinh Loc
Lieng The Phy
Lam Le Thanh The

15CTT - APCS - HCMUS

CS414 - Machine Learning
Final Project - Handwriting Recognition for Basic Arithmetic Expressions

Dec 2018
"""

import os
import numpy as np
import cv2
from multiprocessing import Pool
import _pickle as pickle
import time
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from sklearn.preprocessing  import LabelEncoder

class DatasetHelper:
    @staticmethod
    def remove_unnecessary_files(folder, limit_size):
        label_folders = os.listdir(folder)
        for label_folder in label_folders:
            if label_folder != "div":
                folder_path = '\\'.join([folder, label_folder])
                images = os.listdir(folder_path)
                delete_images = images[limit_size:]
                for image in delete_images:
                    os.remove('\\'.join([folder_path, image]))


class Dataset:

    def __init__(self, dataset_folder_path, test_size=0.3):
        self.dataset_folder_path = dataset_folder_path
        self.labels = os.listdir(self.dataset_folder_path)
        self.test_size = test_size

    def get_label_folder_paths(self):
        return np.array(["/".join([self.dataset_folder_path, label])
                         for label in self.labels])

    def get_image_as_numpy_array(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image.flatten()

    def get_folder_from_label_name(self, label):
        return "/".join([self.dataset_folder_path, label])

    def get_images_in_a_folder(self, folder_path):
        list_images = os.listdir(folder_path)
        result = []

        for image in list_images:
            result.append(self.get_image_as_numpy_array("\\".join([folder_path, image])))

        return result

    def get_trainning_set(self, label):
        data = []

        label_folder = self.get_folder_from_label_name(label)
        list_images = self.get_images_in_a_folder(label_folder)
        data = data + list_images

        return data

    def get_all_training_sets(self, number_of_processes=5):
        p = Pool(processes=number_of_processes)
        result = p.map(self.get_trainning_set, self.labels)
        p.close()
        p.join()

        data_train = []
        data_test = []
        label_train = []
        label_test = []

        for i in range(len(self.labels)):
            target = [self.labels[i]] * len(result[i])
            x_train, y_train, x_test, y_test = train_test_split(result[i], target, test_size=self.test_size)
            data_train += x_train
            label_train += y_train
            data_test += x_test
            label_test += y_test
            
        return np.array(data_train), np.array(data_test), np.array(label_train), np.array(label_test)


class TrainingModel:

    def __init__(self, model, data, target):
        self.model = model
        self.data = data
        self.target = target

    def train_model(self):
        self.model.fit(self.data, self.target)

    def save(self, name):
        with open(name, 'wb') as fp:
            pickle.dump(self.model, fp)
        print("saved")

class CNN(TrainingModel):
    def __init__(self,model,data,target,n_classes):
        TrainingModel.__init__(self,model,data,target)
        self.n_classes = n_classes
    def define_mode(self):
        self.model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

        self.model.add(Conv1D(32, 9, input_shape=(2025,1), activation='relu'))
        self.model.add(MaxPooling1D())

        self.model.add(Conv1D(64, 9, activation='relu'))
        self.model.add(MaxPooling1D())
        
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        
    def train_model(self,test_data,test_target):
        self.define_mode()
        history  = self.model.fit(self.data,self.target,batch_size =64,epochs = 50,verbose= 1,validation_data= (test_data,test_target) )
        return history
    def save(self,name):
        save_dir ="./"
        model_path = os.path.join(save_dir,name)
        self.model.save(model_path)

class ModelEvaluation:

    def __init__(self, model_path, data, target):
        self.model = self.load_model(model_path)
        self.test_data = data
        self.target = target

    def evaluate(self):
        predicted = self.model.predict(self.test_data)
        print("Classification report for classifier %s:\n%s\n" %
              (self.model, metrics.classification_report(self.target, predicted)))
        print(self.target)
        print("predicted: 	", predicted)

    def load_model(self, path):
        with open(path, 'rb') as fp:
            clf = pickle.load(fp)
        return clf
class CNNEvaluation(ModelEvaluation):
    def evaluate(self):
        loss_metrics = self.model.evaluate(self.test_data,self.target)
        print("Test Lost: ", loss_metrics[0])
        print("Test Accuracy: ",loss_metrics[1])

def get_image_as_numpy_array(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	return image.flatten()

if __name__ == "__main__":
    pass
    test1= "data"
    n_classes = 14
    dataset = Dataset(test1)
    X_train, y_train, X_test, y_test = dataset.get_all_training_sets()
    print(X_train.shape)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train/=255
    X_test/=255
    X_train = np.expand_dims(X_train, axis = 2)
    X_test = np.expand_dims(X_test, axis = 2)
    label_encoder = LabelEncoder()
    ytrain_encoder = label_encoder.fit_transform(y_train)
    ytest_encoder = label_encoder.fit_transform(y_test)
    y_train = np_utils.to_categorical(ytrain_encoder,n_classes)
    y_test = np_utils.to_categorical(ytest_encoder,n_classes)

    print("Training model...")
    CNN_model = CNN(Sequential(),X_train,y_train,n_classes)
    CNN_model.train_model(X_test,y_test)
    CNN_model.save('cnn_model.h5')

    input('FINISHED...')
