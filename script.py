import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import datetime
from tensorflow.python.framework.errors_impl import  InvalidArgumentError
from colorama import Fore, Style
import argparse
import os


 
def main():
	parser = argparse.ArgumentParser(description="TensorFlow Training Pipeline")
	parser.add_argument('--data_root', type=str, help='root folder of the image dataset')
	#parser.add_argument('--save_model_dir', type=str, help='directory to save the  trained model')
	args = parser.parse_args()
	
	if(args.data_root==None):
		print(Fore.RED + "--data_root= specify the root folder of the image dataset")
		print(Style.RESET_ALL)  
		exit()
	
		        
	mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
	

	feature_extractor_model = mobilenet_v2

	feature_extractor_layer = hub.KerasLayer(
	    feature_extractor_model,
	    input_shape=(224, 224, 3),
	    trainable=False)


	batch_size = 32
	img_height = 224
	img_width = 224

	try:
	  train_ds = tf.keras.utils.image_dataset_from_directory(
	  str(args.data_root),
	  validation_split=0.2,
	  subset="training",
	  seed=123,
	  image_size=(img_height, img_width),
	  batch_size=batch_size
	  )

	  val_ds = tf.keras.utils.image_dataset_from_directory(
	  str(args.data_root),
	  validation_split=0.2,
	  subset="validation",
	  seed=123,
	  image_size=(img_height, img_width),
	  batch_size=batch_size
	  )
	except ValueError:
	  print(Fore.GREEN + "'\n' Follow this directory structure '\n' main_directory/'\n' ...class_a/ '\n' ......a_image_1.jpg '\n' ......a_image_2.jpg '\n' ...class_b/ '\n' ......b_image_1.jpg '\n' ......b_image_2.jpg")
	  print(Style.RESET_ALL)  
	  exit()

	class_names = np.array(train_ds.class_names)
	print(class_names)

	normalization_layer = tf.keras.layers.Rescaling(1./255)
	train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
	val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



	for image_batch, labels_batch in train_ds:
	  print(image_batch.shape)
	  print(labels_batch.shape)
	  break

	feature_batch = feature_extractor_layer(image_batch)
	print(feature_batch.shape)

	num_classes = len(class_names)

	model = tf.keras.Sequential([
	  feature_extractor_layer,
	  tf.keras.layers.Dense(num_classes)
	])

	model.summary()

	predictions = model(image_batch)

	predictions.shape

	model.compile(
	  optimizer=tf.keras.optimizers.Adam(),
	  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	  metrics=['acc'])

	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(
	    log_dir=log_dir,
	    histogram_freq=1) # Enable histogram computation for every epoch.

	NUM_EPOCHS = 10

	try:
	  history = model.fit(train_ds,
			    validation_data=val_ds,
			    epochs=NUM_EPOCHS,
			    callbacks=tensorboard_callback)
	except InvalidArgumentError:
	  print(Fore.RED +"\n'Unknown image file format identified/images corrupted. One of JPEG, PNG, GIF, BMP required.")
	  print(Style.RESET_ALL)

	t = time.time()

	save_model_dir="trained_model/"
	export_path = save_model_dir.format(int(t))
	print(Fore.GREEN + "trained model saved in "+ os.path.abspath(save_model_dir))
	print(Style.RESET_ALL)  
	
	model.save(export_path)

	export_path

if __name__ == "__main__":
	main()

