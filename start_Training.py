import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Build the preloader array, resize images to 128x128
from tflearn.data_utils import image_preloader
from tflearn.metrics import R2

#%%
import numpy as np
from numpy.random import randint

#%%

def defineArchitecture():

    # Input is a 32x32 image with 3 color channels (red, green and blue)
    network = convnet = input_data(shape=[None, 64, 64, 1], name='input')

    # Step 1: Convolution
    network = conv_2d(network, 32, 3, activation='relu')

    # Step 2: Max pooling
    network = max_pool_2d(network, 2)

    # Step 3: Convolution again
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 4: Convolution yet again
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 5: Max pooling again
    network = max_pool_2d(network, 2)

    # Step 6: Fully-connected 512 node neural network
    network = fully_connected(network, 512, activation='relu')

    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    network = dropout(network, 0.5)

    network = fully_connected(network, 1, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=0.01, loss='mean_square', metric=R2(), name='targets')

    return network
#%%


model = tflearn.DNN(defineArchitecture())

train_dataset_file = 'corpus-da.txt'

train_GT_file = './corpus-da/datadescription.txt'

with open(train_GT_file) as inFile:
	list = []
	for line in inFile:
		parts = line.split('\t')       
		list.append([float(parts[4])])
	Y = np.array(list)

X, _ = image_preloader(train_dataset_file, image_shape=(64, 64),   mode='file', categorical_labels=False, normalize=False, grayscale=True)
X = np.reshape(X, (-1, 64, 64, 1))

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=0.1, snapshot_epoch=True, snapshot_step=500, show_metric=True, run_id='da-simulated')

model.save('da.model')



