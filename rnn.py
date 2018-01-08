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

top_words = 5000

#Limiting the maxlen to 250 since most articles have lesser than 250 length
(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=top_words,
                                                         maxlen=250,
                                                         test_split=0.2)
x_train = sequence.pad_sequences(x_train, maxlen=250)
x_test = sequence.pad_sequences(x_test, maxlen=250)
from keras.utils import to_categorical

y_train_en = to_categorical(y_train)
y_test_en = to_categorical(y_test)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=250))
model.add(LSTM(100))
model.add(Dense(46))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train_en, validation_data=(x_test, y_test_en), epochs=5, batch_size=128)

#Done, can be ran in notebook or here
