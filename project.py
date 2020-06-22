import os
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import zipfile
from skimage.io import imread
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def baseline_model():
    # the function creates a model
    # output: a model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(200, 200, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (5, 5), input_shape=(200, 200, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (5, 5), input_shape=(200, 200, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5, 5), input_shape=(200, 200, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile model
    return model


def train_model(model, save_file_path, dir_path):
    # input: a model, path to save the model in, dataset path.
    # the function trains a model with the dataset and save it.
    model_wanted_name = input("type the wanted name of the model: ")
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, shear_range=0.2, zoom_range=0.2, rotation_range=20,
                                 horizontal_flip=True)
    # prepare iterator
    train_it = datagen.flow_from_directory(os.path.join(dir_path, "Train"),
                                           class_mode='categorical', batch_size=16, target_size=(200, 200))
    test_it = datagen.flow_from_directory(os.path.join(dir_path, "Test"),
                                          class_mode='categorical', batch_size=16, target_size=(200, 200))
    # fit model
    info = model.fit_generator(train_it, steps_per_epoch=len(train_it) // 16, validation_data=test_it,
                               validation_steps=len(test_it) // 16, epochs=3, verbose=1)
    # evaluate model
    loss, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)

    print('> accuracy: {:06.3f}'.format(float(acc) * 100.0))
    model.save(os.path.join(save_file_path, model_wanted_name))
    return info


def zip_files():
    # the function checks if the user wants to extract zip files.
    # if the user chooses to extract files, the function gets the path to the zip file and checks that
    # the path is valid and a zipped one.
    # the function gets a directory from the user that he wants to extract the zip files to. after checking that the
    # input is valid the function gets a name from the user to the new file and extracts the zip file to the wanted
    # directory.
    # if the user chooses to not extract files, the function gets the path of the dataset and checks that the path
    # is valid.
    # output: the function returns the path to the dataset.
    while True:
        user_operation = input("do you want to extract zipped dataset?\n1. yes\n2. no\n"
                               "type the number of the wanted operation: ")
        # user operation, to extract zip files or not.
        if user_operation == "1":  # the user wants to extract zip files
            while True:
                path_to_zip_file = input("type zip file path: ")
                # path to the zip dataset that the user wants to extract
                if not os.path.isfile(path_to_zip_file) or not path_to_zip_file.endswith(".zip"):
                    # there is no such file or not a zip file
                    print("file does not exist or not a zip file")
                else:  # there is a zip file
                    while True:
                        directory_to_extract_to = input("type directory to extract to: ")
                        # the directory the user wants to extract the zipped dataset file to
                        if not os.path.isdir(directory_to_extract_to):  # there is no such directory
                            print("directory does not exist")
                        else:
                            wanted_name = input("type the wanted name of the new file: ")
                            # user wanted name to the new file that was extracted
                            print("extracting files...")
                            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                                zip_ref.extractall(os.path.join(directory_to_extract_to, wanted_name))
                                # extract the zip file to the directory and giving the wanted name
                                print("finished extracting files")
                                print("\n")
                            return os.path.join(directory_to_extract_to, wanted_name)
                            # return path to the extracted files
        elif user_operation == "2":  # doesn't want to extract files
            while True:
                dateset_path = input("type path to dataset: ")  # path to the dataset
                if os.path.isdir(dateset_path):  # there is such directory
                    print("\n")
                    return dateset_path  # return the dataset path
                else:  # there is no such file
                    print("there is no such file")
        else:
            print("input error")


def loading_model():
    # trying to load a model.
    # output: a model
    while True:
        path_to_model_file = input("type model file path: ")  # path to the model
        if not os.path.isfile(path_to_model_file):  # there is no such model file
            print("model file does not exist")
        else:
            try:  # if its a model
                model = load_model(path_to_model_file)
            except:  # the file is not a model
                print("file is not a model")
                continue  # return to the while True
            try:  # looking for a model that fit to the project and Keras can work with
                path_to_pic = r'C:\Users\User\Documents\DL\DataSet\Train\black\0.jpg.chip.jpg'  # a valid pic from the
                # dataset to check the model on.
                pic = imread(path_to_pic)
                pic = pic.astype('float32')/255  # model get a float32 pic
                pic = resize(pic, (200, 200), mode='constant', preserve_range=True)
                pic = np.expand_dims(pic, axis=0)  # turn an image to a tensor 4D
                model.predict(pic)
                return model
            except:
                print("model is not compatible")


def prediction(model):
    # input: a model
    # the function gets a picture from the user and checks that it's a valid one.
    # the function loads the image and makes changes on it to make sure that the picture is compatible to the project.
    # the function predicts the skin color of the human according to the model.
    # the function prints the prediction of what is the color of the human in the given image.

    while True:
        path_to_pic = input("type wanted picture to predict: ")  # path to a picture
        if not os.path.isfile(path_to_pic):  # there is no such picture file
            print("picture file does not exist")
        else:
            try:
                pic = imread(path_to_pic)  # method loads the image from the given path.
                pic = pic.astype('float32')  # change the picture type. model get a float32 pic
                pic = resize(pic, (200, 200), mode='constant', preserve_range=True)/255
                # resize to 200X200. preserve similar ratio, keep the original range of values as a numpy array.
                # divided by 255, normalization to 0-1 values.
                pic = np.expand_dims(pic, axis=0)  # turn an image to a tensor 4D
                pred = model.predict(pic)  # array of 2 values: percentage of black and percentage of white
                #  [[0.07883145, 0.9211686 ]] [percentage of black, percentage of white]
                predicted_class_indices = np.argmax(pred, axis=1)  # bigger percentage index 0->black index 1->white
                labels = {0: 'black', 1: 'white'}  # dictionary labels of categories
                predictions = labels[int(predicted_class_indices)]  # category of the predicted picture. int of the
                # bigger value means the index in the dictionary. label of the index.
                print("the human's face skin color is: " + str(predictions))
                print("\n")
                break
            except:
                print("file is not a compatible image")


def graphs(info):
    # input: model information for each epoch
    # the function prints accuracy and loss graphs. the function gives an option to save the graphs.
    while True:
        want_to_save = input("would you like to save the graphs?\n1. yes\n2. no\n"
                             "type the number of the wanted operation: ")
        # user wants to save or not, 1 is yes, 2 is no
        if want_to_save == "1":
            while True:
                directory_to_save_graphs = input("input directory to save the graphs in: ")
                if os.path.isdir(directory_to_save_graphs):  # there is  such directory
                    print("directory is ok")
                    break
                else:
                    print("directory does not exist")
        if want_to_save == "1":
            break
        if want_to_save == "2":
            break
        else:
            print("input error")

    history = info.history  # dictionary with  info metrics(acc, val acc, loss, val loss) get from model training
    for key in history.keys():
        if "acc" in key:  # accuracy graph
            plt.plot(history[key], label=key)
    plt.legend()
    if want_to_save == "1":
        print("saving accuracy graph")
        plt.savefig(os.path.join(directory_to_save_graphs, input("name of accuracy graph: ")))
    plt.show()

    for key in history.keys():
        if "loss" in key:  # loss graph
            plt.plot(history[key], label=key)
    plt.legend()
    if want_to_save == "1":
        print("saving accuracy graph")
        plt.savefig(os.path.join(directory_to_save_graphs, input("name of loss graph: ")))
    plt.show()
    print("\n")


def main():
    dataset_path = zip_files()
    while True:
        user_operation = input(" What Operation do you want to take?\n1. train model\n2. predict picture\n3. exit\n "
                               "type the number of the wanted operation: ")
        if user_operation == "1":
            while True:
                model_operation = input(
                    "would you like to load or to create a model?\n1. create a model\n2. load a model\n"
                    "type the number of the wanted operation: ")
                if model_operation == "1":
                    model = baseline_model()  # model created in the script
                    break
                elif model_operation == "2":
                    model = loading_model()
                    break
                print("input error")
            while True:
                wanted_model_path = input("enter path to save your model: ")
                if os.path.isdir(wanted_model_path):  # there is such directory
                    print("directory is ok")
                    break
                else:
                    print("directory does not exist")
            information = train_model(model, wanted_model_path, dataset_path)
            graphs(information)
        elif user_operation == "2":
            model = loading_model()  # loaded from file
            prediction(model)
        elif user_operation == "3":
            sys.exit()
        else:
            print("input error")


if __name__ == '__main__':
    main()
