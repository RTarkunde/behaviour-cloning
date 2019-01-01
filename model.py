################################################################################
# Behaviour learning Self Driving Car Model Simulator Trainer
# Written for KPIT Self Learning Car Driving model Project 4 assignment
# Author: Rahul Tarkunde
# 29 th Dec, 2018
################################################################################

import csv
import cv2
import numpy as np

################################################################################
# The training data consists of two parts.
# 1 : Udacity suplied training data in folder  ./SimulatorData/UdacityData
# 2 : Project Generated data /SimulatorData/ProjectData/data/
# The following flag switches the Project data for use  for training and
# validation
use_project_data = True
################################################################################

udacity_data_lines = []
project_data_leftb_lines = []
project_data_rightb_lines = []
project_data_bridge_lines = []
################################################################################
# Function: Load the simulator data from the csv file
################################################################################
def readCSVFile(path, lines):
    j     = 0
    with open (path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            if j != 0: # Skip the first line for header
                 lines.append(line)
            j += 1
        csvfile.close()
####

readCSVFile('./SimulatorData/UdacityData/data/driving_log.csv' ,udacity_data_lines)

################################################################################
# The project data is again divided in two types.
# Road side  recovery from Left Bank and Right Bank.
# /LeftBank and /RightBank
################################################################################

if use_project_data == True:
    readCSVFile('./SimulatorData/ProjectData/data/LeftBank/driving_log.csv'  , project_data_leftb_lines)
    readCSVFile('./SimulatorData/ProjectData/data/RightBank/driving_log.csv' , project_data_rightb_lines)
    readCSVFile('./SimulatorData/ProjectData/data/bridge/driving_log.csv' , project_data_bridge_lines)
##

print('Total number of images')
print (np.shape(udacity_data_lines) + np.shape(project_data_leftb_lines) + np.shape(project_data_rightb_lines))

images       = []
measurements = []
filename     = ''

################################################################################
# Arbitary trial correction factor applied to left and right camera images
correction   = 0.4

for line in udacity_data_lines:
        source_path  = line[0]
        trackname    = 'UdacityData'
        driveType    = 'data'

        ########################################################################
        #  Process center camera images
        cfilename     = source_path.split('/')[-1] # Center Camera
        # The c prefix for filename variable as it holds image names for center
        # camera
        current_path = './SimulatorData/' + trackname + '/' + driveType + '/IMG/' + cfilename
        # Process Steering angle
        measurement = float(line[3])
        # The following check ignores all the images and entries which has steering
        # angle close to 0 that is  -0.01 <= 0 <- 0.01
        if measurement < -0.01 or measurement > 0.01:
            image        = cv2.imread(current_path)
            images.append(image)
            measurements.append(measurement)
        ###

        ########################################################################
        #  Process left camera images#
######## Ignore left camera images to keep the training speed performance high
        lfilename     = source_path.split('/')[-1] # Left Camera
        current_path  = './SimulatorData/' + trackname + '/' + driveType + '/IMG/' + lfilename
        if measurement < -0.01 or measurement > 0.01:
            image        = cv2.imread(current_path)
            images.append(image)
            measurement  = float(line[3]) + correction
            measurements.append(measurement)

        ########################################################################
        # Process Right camera images#
        # 2 because the right  camera entries are third in the csv files
        source_path  = line[2]
        rfilename     = source_path.split('/')[-1] # Right Camera
        current_path = './SimulatorData/' + trackname + '/' + driveType + '/IMG/' + rfilename
        if measurement < -0.01 or measurement > 0.01:
            image        = cv2.imread(current_path)
            images.append(image)
            measurement  = float(line[3]) - correction
            measurements.append(measurement)
        ###
###
################################################################################
print('Average of Udacity data steering angle')
print(np.average(measurements))

################################################################################
# Using the data generated on local machine for more training.
if use_project_data == True:
    ############################################################################
    # Left bank recovery data.
    for line in project_data_leftb_lines:
        source_path  = line[0]
        trackname    = 'ProjectData'
        driveType    = 'data'
        cfilename    = source_path.split('/')[-1] # Center Camera
        ########################################################################
        #  Process center camera images
        current_path = './SimulatorData/' + trackname + '/' + driveType + '/LeftBank/IMG/' + cfilename
        image        = cv2.imread(current_path)
        images.append(image)
        # Steering angle
        measurement  = float(line[3])
        measurements.append(measurement)
    ###

    ############################################################################
    # Right bank recovery data.
    for line in project_data_rightb_lines:
        source_path  = line[0]
        trackname    = 'ProjectData'
        driveType    = 'data'
        cfilename    = source_path.split('/')[-1] # Center Camera
        ########################################################################
        #  Process center camera images
        current_path = './SimulatorData/' + trackname + '/' + driveType + '/RightBank/IMG/' + cfilename
        image        = cv2.imread(current_path)
        images.append(image)
        # This is the steering angle
        measurement  = float(line[3])
        measurements.append(measurement)
    ###
    ############################################################################
    # Bridge recovery data.
    for line in project_data_bridge_lines:
        source_path  = line[0]
        trackname    = 'ProjectData'
        driveType    = 'data'
        cfilename    = source_path.split('/')[-1] # Center Camera
        ########################################################################
        #  Process center camera images
        current_path = './SimulatorData/' + trackname + '/' + driveType + '/bridge/IMG/' + cfilename
        print(current_path)
        image        = cv2.imread(current_path)
        images.append(image)
        # This is the steering angle
        measurement  = float(line[3])
        measurements.append(measurement)
    ###

###

print('Average of project generated data steering angle')
print(np.average(measurements))

################################################################################
## Augmenting of all the images flipping them
################################################################################
augmented_images, augmented_measurements = [], []
for image, measurement in zip (images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    # Flip the image
    augmented_images.append(cv2.flip(image, 1))
    # "Flip" the steering or else it will be runaway vehicle
    augmented_measurements.append(measurement * -1.0)
###

print('Number of images after augmentation')
print(np.shape(augmented_images))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Cropping2D
from keras.layers import Dropout
from keras.layers.core import K

################################################################################
## Intial Test Train model
################################################################################
"""
model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160 , 320 , 3)))
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Flatten()) #input_shape=(160, 320, 3)))
model.add(Dense(1))
"""

################################################################################
## NVidia Self Driving Car Based Training Model Pipeline
################################################################################
dropout_rate = 0.025
model        = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160 , 320 , 3)))
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Conv2D(24, (5, 5), activation = "relu", strides=(2, 2)))
model.add(Dropout(dropout_rate))
model.add(Conv2D(36, (5, 5), activation = "relu", strides=(2, 2)))
model.add(Dropout(dropout_rate))
model.add(Conv2D(48, (5, 5), activation = "relu", strides=(2, 2)))
model.add(Dropout(dropout_rate))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(Dropout(dropout_rate))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(Dropout(dropout_rate))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#####

################################################################################
# We use mse ( mean square error) loss function instead of cross entropy
# because this is a regression network instead of a lassification network.
model.compile(loss='mse', optimizer='adam')
# 20% of the images are reserved for validation.
# Train for 5 interations
model.fit(X_train, y_train, validation_split=0.2, shuffle = True, epochs = 4)


################################################################################
# Save the kerase model snapshot in current directory.
# This will be used in autonomous driving mode.
model.save('model.h5')
################################################################################


################################################################################
# Following code loads the keras model and data. Not used
################################################################################
#from keras.models import load_model
#model.save('model.h5')
#del model  # deletes the existing model

################################################################################
# returns a compiled model
################################################################################
#identical to the previous one
#model = load_model('model.h5')
