'''
Demonstration of the power of RNN with CNN.
Basic classifier using the Conv-LSTM architecture. Reuters data set is available with keras library
The accuracy can be further tuned by using the pretrained embedding matrix and some data cleaning.
'''

# Legendary imports
import numpy as np
from keras.datasets import imdb
from keras.layers.core import Dense, Dropout, Activation
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

# Restrict the size of vocab to 5000
top_words = 5000


#Limiting the maxlen to 250 since most articles have lesser than 250 length
(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=top_words,
                                                         maxlen=250,
                                                         test_split=0.2)

# Since keras LSTM don't support variable input, so we will have to pad the sequences.

x_train = sequence.pad_sequences(x_train, maxlen=250)
x_test = sequence.pad_sequences(x_test, maxlen=250)
from keras.utils import to_categorical

y_train_en = to_categorical(y_train)
y_test_en = to_categorical(y_test)


# Create the model using keras Sequential api
embedding_vecor_length = 64
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=250))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(200))
model.add(Dense(100))
model.add(Dense(46))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(x_train, y_train_en, validation_data=(x_test, y_test_en), epochs=5, batch_size=64)

#Done, can be ran in notebook or here
# Without CNN, output was 63% accuracy
# With CNN, output is 
# 67% accuracy in 46 classes is really good
''' Here is the output of the model
Train on 7602 samples, validate on 1901 samples
Epoch 1/5
7602/7602 [==============================] - 89s - loss: 2.0788 - acc: 0.4559 - val_loss: 1.7150 - val_acc: 0.5850
Epoch 2/5
7602/7602 [==============================] - 87s - loss: 1.5937 - acc: 0.5710 - val_loss: 1.6624 - val_acc: 0.5634
Epoch 3/5
7602/7602 [==============================] - 81s - loss: 1.4152 - acc: 0.6251 - val_loss: 1.4828 - val_acc: 0.6376
Epoch 4/5
7602/7602 [==============================] - 81s - loss: 1.2608 - acc: 0.6711 - val_loss: 1.4203 - val_acc: 0.6633
Epoch 5/5
7602/7602 [==============================] - 81s - loss: 1.1208 - acc: 0.7059 - val_loss: 1.4299 - val_acc: 0.6707
'''



