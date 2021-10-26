from numpy.random import seed
seed(8) #1

import tensorflow
tensorflow.random.set_seed(7)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

import os
import argparse
import glob

from azureml.core import Run
import azureml
from azureml.core import Workspace, Datastore, Dataset
from azureml.core import Experiment

import keras
from keras.models import Sequential, model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import models
from tensorflow.python.keras import layers

from tensorflow.keras import optimizers





parser = argparse.ArgumentParser()
parser.add_argument('--train-data-folder', type=str, dest='train_data_folder', default='train', help='train data folder mounting point')
parser.add_argument('--test-data-folder', type=str, dest='test_data_folder', default='test', help='test data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=10, help='mini batch size for training')
parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=5, help='learning ratenumber of epochs')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')



args = parser.parse_args()




DATASET_PATH  = args.train_data_folder
TEST_DIR      = args.test_data_folder
IMAGE_SIZE    = (150, 150)
NUM_CLASSES   = 4
BATCH_SIZE    = args.batch_size # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = args.num_epochs
LEARNING_RATE = args.learning_rate


print('train data path: ',DATASET_PATH)
print('test data path: ',TEST_DIR)
print("batch",BATCH_SIZE)
print("NUM_EPOCHS",NUM_EPOCHS)



#Data augmentation is the process of generating new images from a dataset with random modifications. It results in a better deep learning model 
#it is mostly important for small datasets.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=50,
                                   featurewise_center = True,
                                   featurewise_std_normalization = True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.25,
                                   zoom_range=0.1,
                                   #zca_whitening = True,
                                   channel_shift_range = 20,
                                   horizontal_flip = True ,
                                   vertical_flip = True ,
                                   validation_split = 0.2,
                                   fill_mode='constant')

train_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "training",
                                                  seed=42,
                                                  class_mode="categorical"   #For multiclass use categorical n for binary use binary
                                                  )

valid_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "validation",
                                                  seed=42,
                                                  class_mode="categorical"  #For multiclass use categorical n for binary use binary
                                                 
                                                  )

#Simple CNN model based on Xception. Set dense layer neuron count same as the no. of output classes 
#If you wnna use a saved model then skip this step


from tensorflow.keras.applications import Xception

conv_base = Xception(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

conv_base.trainable = True

model = models.Sequential()
model.add(conv_base)


model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',     #for multiclass use categorical_crossentropy
              
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              metrics=['acc'])

# start an Azure ML run
run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['val_loss'])
        run.log('Accuracy', log['val_accuracy'])

print(len(train_batches))
print(len(valid_batches))
#NUM_EPOCHS    = 50

STEP_SIZE_TRAIN=train_batches.n//train_batches.batch_size
STEP_SIZE_VALID=valid_batches.n//valid_batches.batch_size

history=model.fit(train_batches,
                        steps_per_epoch =STEP_SIZE_TRAIN,
                        validation_data = valid_batches,
                        validation_steps = STEP_SIZE_VALID,
                        epochs= NUM_EPOCHS,
                        verbose=2,
                        callbacks=[LogRunMetrics()]
                       )

test_datagen = ImageDataGenerator(rescale=1. / 255)
#TEST_DIR =  'covid-19/four_classes/test'
eval_generator = test_datagen.flow_from_directory(TEST_DIR,target_size=IMAGE_SIZE,batch_size=1, 
                                                  shuffle=False, seed=42, class_mode="categorical")
#eval_generator.reset()

eval_generator.reset()  
x = model.evaluate(eval_generator,
                           steps = np.ceil(len(eval_generator)), 
                           use_multiprocessing = False,
                           verbose = 1,
                           workers=1,
                           )


print('Test loss:' , x[0])
print('Test accuracy:',x[1])

# log a single value
run.log("Final test loss", x[0])
print('Test loss:', x[0])

run.log('Final test accuracy', x[1])
print('Test accuracy:', x[1])

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")
