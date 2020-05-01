import keras

ACCURACY_THRESHOLD = 0.5

class AccHistory(keras.callbacks.Callback):
	def on_batch_end(self, batch, logs={}):
		if(logs.get('acc') > ACCURACY_THRESHOLD):
			self.model.stop_training = True