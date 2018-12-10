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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class DatasetHelper:
    @staticmethod
    def remove_unnecessary_files(folder, limit_size):
        """
        :param folder: name of the folder
        :param limit_size: the number of files in each folder

        Make each folder to have the exact number of files
        """
        label_folders = os.listdir(folder)
        for label_folder in label_folders:
            if label_folder != "div":
                folder_path = '/'.join([folder, label_folder])
                images = os.listdir(folder_path)
                delete_images = images[limit_size:]
                for image in delete_images:
                    os.remove('/'.join([folder_path, image]))


class Dataset:

    def __init__(self, dataset_folder_path, test_size=0.3):
        """
        :param dataset_folder_path: the path to the dataset folder
        :param test_size:  the test size. The default is 20% over the dataset
        """
        self.dataset_folder_path = dataset_folder_path
        self.labels = os.listdir(self.dataset_folder_path)
        self.test_size = test_size

    def get_label_folder_paths(self):
        """
        :return: The full path to each label folder
        """
        return np.array(["/".join([self.dataset_folder_path, label])
                         for label in self.labels])

    def get_image_as_numpy_array(self, image_path):
        """
        This function get the image and convert it to numpy array
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image.flatten()

    def get_folder_from_label_name(self, label):
        return "/".join([self.dataset_folder_path, label])

    def get_images_in_a_folder(self, folder_path):
        """
        :param folder_path: path to the folder
        :return: load all images in a folder as numpy array
        """
        list_images = os.listdir(folder_path)
        result = []

        for image in list_images:
            result.append(self.get_image_as_numpy_array("/".join([folder_path, image])))

        return result

    def get_trainning_set(self, label):
        """
        :param label: Label name
        :return: Images as numpy array in the label folder
        """
        data = []

        label_folder = self.get_folder_from_label_name(label)
        list_images = self.get_images_in_a_folder(label_folder)
        data = data + list_images

        return data

    def get_all_training_sets(self, number_of_processes=5):
        """
        :param number_of_processes: The number of processes used for loading the dataset
        :return: Images as numpy array of all folder with label
        """
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
        """
        :param model: types of model
        :param data: data to train
        :param target: label for the data

        """
        self.model = model
        self.data = data
        self.target = target

    def train_model(self):
        self.model.fit(self.data, self.target)

    def save(self, name):
        with open(name, 'wb') as fp:
            pickle.dump(self.model, fp)
        print("saved")


class GridSearchTraining(TrainingModel):

    def __init__(self, model, tuned_parameters, data, target):
        """
               :param model: types of model
               :param tuned_parameters: parameters for tuning
               :param data: data to test
               :param target: label for the data

               """
        super(GridSearchTraining, self).__init__(GridSearchCV(model, tuned_parameters, cv=5, n_jobs=-1),
                                                 data,
                                                 target
                                                 )

    def save(self, name):
        print("Best parameters:")
        print(self.model.best_params_)
        with open(name, 'wb') as fp:
            pickle.dump(self.model.best_estimator_, fp)
        print("saved")


class ModelEvaluation:

    def __init__(self, model_path, data, target):
        """
        :param model_path: path to the model
        :param data: data to test
        :param target: label for the data

        """
        self.model = self.load_model(model_path)
        self.test_data = data
        self.target = target

    def evaluate(self):
        predicted = self.model.predict(self.test_data)
        print("Classification report for classifier %s:\n%s\n" %
              (self.model, metrics.classification_report(self.target, predicted)))
        print(self.target)
        print(predicted)

    def load_model(self, path):
        with open(path, 'rb') as fp:
            clf = pickle.load(fp)
        return clf


if __name__ == "__main__":
    #Load the dataset
    print("Load dataset...")
    t = time.time()
    dataset = Dataset("data/data")
    X_train, y_train, X_test, y_test = dataset.get_all_training_sets()
    print(X_train.shape)
    print("Time elapsed: %d", time.time() - t)

    # Set the parameters by cross-validation
    tuned_parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}, ]
    #Using gridsearchCV to find optimal parameters for the model
    print("Training model...")
    t = time.time()
    model = GridSearchTraining(SVC(), tuned_parameters, X_train, y_train)
    model.train_model()
    model.save("svm_model.mdl")
    print("Time elapsed: %d", time.time() - t)

    #Evaluate the model
    t = time.time()
    print("Evaluate model...")
    model_evaluation = ModelEvaluation("svm_model.mdl", X_test, y_test)
    model_evaluation.evaluate()
    print("Time elapsed: %d", time.time() - t)
