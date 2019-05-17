import numpy as np
from PIL import Image
import os
import cv2
import imageio
import inspect
import glob
import keras
import keras.layers as L 
import keras.backend as K
import tensorflow as tf
# import progressbar
import sys
import shutil
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split

def genData(period_length, samples):
	x = np.linspace(0, period_length, samples)
	y = np.random.randn(x.shape[0])
	return x, y

def genPolyData(maxPolyDegree, polyWeightVariance, window_x):
	degree = np.random.randint(maxPolyDegree+1)
	weights = np.random.randn(degree)*polyWeightVariance
	generatedPoly = np.poly1d(weights)

	return generatedPoly(window_x)


def genSinglePeriod(period_length, window_size, index, maxPolyDegree, polyWeightVariance):
	import matplotlib
	matplotlib.use('agg')
	import matplotlib.pylab as plt
	
	samples = 5
	x, y = genData(period_length, samples)
	image_height = 3
	image_width = 10
	window_x = x
	window_y = y
	for i in range(int(window_size/period_length-1)):
		window_x = np.concatenate((window_x, x + window_x[-1]))
		window_y = np.concatenate((window_y, y))
	# if window_x.shape[0] < samples * window_size:
	# 	window_x = np.concatenate((window_x, (int(window_size/period_length)+1)*x[:(samples * window_size-window_x.shape[0])]))
	# 	window_y = np.concatenate((window_y, y[:(samples * window_size-window_y.shape[0])]))

	window_y += genPolyData(maxPolyDegree, polyWeightVariance, window_x)

	plt.rcParams["figure.figsize"] = (image_width, image_height)
	plt.plot(window_x, window_y)
	plt.grid(False)
	plt.axis('off')
	plt.savefig('rawData/'+ str(index) +'_period_length_'+str(period_length)+'_window_length_'+str(window_size)+'.png', bbox_inches='tight')
	plt.clf()



def getModel(img_shape, number_of_prediction):
	cnn = tf.keras.models.Sequential()

	cnn.add(tf.keras.layers.InputLayer(img_shape))
	cnn.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 1), padding="same"))
	cnn.add(tf.keras.layers.Conv2D(16, kernel_size=(1, 3), padding="same"))
	cnn.add(tf.keras.layers.BatchNormalization(axis=-1))
	cnn.add(tf.keras.layers.LeakyReLU(0.1))
	cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
	cnn.add(tf.keras.layers.Dropout(rate=0.25))

	cnn.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same"))
	cnn.add(tf.keras.layers.Conv2D(32, kernel_size=(1, 3), padding="same"))
	cnn.add(tf.keras.layers.BatchNormalization(axis=-1))
	cnn.add(tf.keras.layers.LeakyReLU(0.1))
	cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
	cnn.add(tf.keras.layers.Dropout(rate=0.25))
	
	cnn.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 1), padding="same"))
	cnn.add(tf.keras.layers.Conv2D(32, kernel_size=(1, 3), padding="same"))
	cnn.add(tf.keras.layers.BatchNormalization(axis=-1))
	cnn.add(tf.keras.layers.LeakyReLU(0.1))
	cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
	cnn.add(tf.keras.layers.Dropout(rate=0.25))
	
	cnn.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 1), padding="same"))
	cnn.add(tf.keras.layers.Conv2D(64, kernel_size=(1, 3), padding="same"))
	cnn.add(tf.keras.layers.BatchNormalization(axis=-1))
	cnn.add(tf.keras.layers.LeakyReLU(0.1))
	cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
	cnn.add(tf.keras.layers.Dropout(rate=0.25))

	cnn.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 1), padding="same"))
	cnn.add(tf.keras.layers.Conv2D(128, kernel_size=(1, 3), padding="same"))
	cnn.add(tf.keras.layers.BatchNormalization(axis=-1))
	cnn.add(tf.keras.layers.LeakyReLU(0.1))
	cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
	cnn.add(tf.keras.layers.Dropout(rate=0.25))

	cnn.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 1), padding="same"))
	cnn.add(tf.keras.layers.Conv2D(128, kernel_size=(1, 3), padding="same"))
	cnn.add(tf.keras.layers.LeakyReLU(0.1))
	cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
	cnn.add(tf.keras.layers.Dropout(rate=0.25))

	# cnn.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same"))
	# cnn.add(tf.keras.layers.LeakyReLU(0.1))
	# cnn.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same"))
	# cnn.add(tf.keras.layers.LeakyReLU(0.1))

	# cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
	# cnn.add(tf.keras.layers.Dropout(rate=0.25))
	cnn.add(tf.keras.layers.Flatten())
	
	cnn.add(tf.keras.layers.Dense(256))
	cnn.add(tf.keras.layers.LeakyReLU(0.1)) 
	cnn.add(tf.keras.layers.Dropout(rate=0.25))
	
	cnn.add(tf.keras.layers.Dense(64))
	cnn.add(tf.keras.layers.LeakyReLU(0.1)) 
	cnn.add(tf.keras.layers.Dropout(rate=0.25))

	cnn.add(tf.keras.layers.Dense(number_of_prediction))
	cnn.add(tf.keras.layers.LeakyReLU(0.1))
	cnn.add(tf.keras.layers.Activation("softmax"))
	return cnn

def experiment(dataset_train, target_train, dataset_test, target_test, epochs,learning_rate,batch_size):
	# reset graph
	K.clear_session()
	# init model 
	cnn = getModel(dataset_train.shape[1:], target_test.shape[-1])
	cnn.summary()

	# training phase
	EPOCHS = epochs
	cnn.compile(
		optimizer=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 
		loss='categorical_crossentropy', 
		metrics=['accuracy']
		)
	
	print('model compiled')
	cnn.fit(
		dataset_train, target_train,
		# batch_size = batch_size,
		epochs = EPOCHS,
		verbose=2,
		validation_data=(dataset_test, target_test)
	)

	score = cnn.evaluate(dataset_test, target_test, batch_size=batch_size)
	print(score)
	K.clear_session()


def paddingImage(fname):
	desired_width = 1000
	desired_height = 250
	im = cv2.imread(fname)
	# print(im.shape)
	old_height = im.shape[0]
	old_width = im.shape[1]

	ratio_width = float(desired_width)/old_width
	ratio_height = float(desired_height)/old_height

	new_width = int(ratio_width*old_width)
	new_height = int(ratio_height*old_height)

	im = cv2.resize(im, (new_width, new_height))
	delta_w = desired_width-new_width
	delta_h = desired_height-new_height
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	color = [0, 0, 0]
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	# print(type(new_im))
	return new_im


def getCurrentLocation():
	return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def readImageByDirectory(dir):
	imageList = []
	labelList = []

	filelist = glob.glob(dir+'/rawData/*.png')
	print("Preparing Image, In totle there are: " + str(len(filelist)) + " images")
	for fname in  filelist:
		tensor = np.empty((250, 250, 3))
		im_padded = paddingImage(fname)
		for i in range(4):
			tensor += im_padded[:, i*250:(i+1)*250]
		
		image = np.array(tensor/4, dtype = np.uint8)
		# image = np.divide(image, 255.0)
		# image = np.subtract(image, 0.5)
		# image = np.multiply(image, 2.0)

		imageList.append(image)
		labelList.append(int(fname.split('_')[-4]))
	# print(labelList)
	return np.array(imageList), keras.utils.to_categorical(labelList)

def draw_stacked_image(path):
	import scipy.misc
	tensor = np.empty((250, 250, 3))
	im_padded = paddingImage(path)
	for i in range(4):
		tensor += im_padded[:, i*250:(i+1)*250]
		
	image = np.array(tensor/4, dtype = np.uint8)
	scipy.misc.imsave('stacked.jpg', image)

	print('we are here')

def drawData(currentLocation, p, size, polyDeg, polyWeight):
	if not os.path.exists(currentLocation+'/rawData'):
		os.makedirs(currentLocation+'/rawData')
	periods = p
	# size = 2000
	# for i in progressbar.progressbar(range(len(periods)*size)):
	for p in periods:
		for i in range(size):
			period_length = p
			window_size = 20
			genSinglePeriod(period_length, window_size, i, polyDeg, polyWeight)
				# bar.update(i*(periods.index(p)+1)
		print("images of periods: " + str(p) + " is drawn")
		sys.stdout.flush()



def CNN(periods=[2,7],sizes=[1000],maxPolyDegrees=[1],polyWeightVariances=[1,2,3],epochs=10,learning_rate=0.0003,batch_size=128,experiments_per_dataset=3):
	for p1 in periods:
		for p2 in periods:
			if p1 < p2:
				for size in sizes:
					for polyDeg in maxPolyDegrees:
						for polyWeight in polyWeightVariances:
							# for epochs in range(10):
							
							# drawData(getCurrentLocation(), [p1, p2], size, polyDeg, polyWeight)

							# tensor = imageio.imread('0_period_length_2_window_length_20.png')
							# tensor = tensor[35:]
							# print(tensor.shape)
							# im = Image.fromarray(tensor, 'RGBA')
							# im.show()
							
							imageList, labelList = readImageByDirectory(getCurrentLocation())
							dataset = imageList
							targets = labelList
							
							dataset_train, dataset_test, target_train, target_test = \
								train_test_split(dataset, targets, test_size=0.1)
							print('dataset_train.shape:' + str(dataset_train.shape))
							print('dataset_train.shape:' + str(dataset_test.shape))
							print('dataset_train.shape:' + str(target_train.shape))
							print('dataset_train.shape:' + str(target_test.shape))
							for i in range(experiments_per_dataset):
								print('!!!  experiment: ' + str(i)+'th  !!!')
								print('Experiment of' + str(size) + 'image of period: '+str(p1)+' vs' + str(size) + 'image of '+str(p2));
								print('polyDeg: ' + str(polyDeg) + ' ; polyWeight: '+str(polyWeight)) 
								# sys.stdout.flush()
								experiment(dataset_train, target_train, dataset_test, target_test, epochs+1,learning_rate,batch_size)
								print('one experiment is done')
								# sys.stdout.flush()							
							# shutil.rmtree('rawData')
	
# if __name__ == '__main__':
# 	folder = 'rawData_10000_2_vs_10000_5_polyWeightVar=4'
# 	image_name = '/1_period_length_2_window_length_20.png'
# 	draw_stacked_image(folder+image_name)
# CNN()




