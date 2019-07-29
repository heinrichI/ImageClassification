import tensorflow.keras

class MySequence(tensorflow.keras.utils.Sequence):
  
    def __init__(self, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.on_epoch_end()

	def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_enzymes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_enzymes) / self.batch_size))