import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(12345)

# The below tf.set_random_seed() will make random number generation in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)

#%% Use GPU or CPU
from keras import backend as K

config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 1} )
sess = tf.Session(config=config)
K.set_session(sess)

#%% Import
import os
import sys
from time import time
import shutil
import datetime
import matplotlib.pyplot as plt
import keras
import keras.utils as ku
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Embedding, LSTM, Dense, Dropout, Add, Input, CuDNNLSTM, GRU
from keras.callbacks import TensorBoard, EarlyStopping
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import Sequential, Model
import glob
import csv

#%%
# general
pad_type = 'pre'
numSamples = 3000 # size of data to train, and to validate
numSamples_val = 0.15*numSamples

# imdb dataset
index_from = 2
maxlen = 200
start_char = 1
num_words = 10000
batch_size = 16

# model params
num_sentiments = 2
total_labels = num_sentiments + 1  # similar to: total_words = len(tokenizer.word_index) + 1
embedding_dim = 100
learning_rate = 0.001
rnn_cell_size = 500
epochs = 100

log_dirname = 'log_file'


#%% LOAD IMDB DATA
# For convenience, words are indexed by overall frequency in the dataset
(x_train_orig, y_train), (x_test_orig, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=num_words, # this is the number of words in vocabulary ( I saw 100000 in theano code) SG
                                                      skip_top=0,
                                                      maxlen=maxlen,
                                                      seed=113,
                                                      start_char=start_char,
                                                      oov_char=2,
                                                      index_from=index_from) # Index actual words with index_from+1 and above
# Meaning of numbers:
# 0 - pad,  1 - start sequence,  2 - unknown,  3 - first word in vocab (most frequent word)

# use labels foe sentiment: 1 and 2 (and not 0 and 1, because 0 is used for pad)
y_train = y_train + 1
y_test = y_test + 1

# reverse lookup
word_to_id = imdb.get_word_index() # "the" is 1, but we want "the" to be 3 like in data
word_to_id = {k:(v+index_from) for k, v in word_to_id.items()}

# add the first three values of the dictionary
word_to_id["<PADsymbol>"] = 0
word_to_id["<STARTsymbol>"] = 1
word_to_id["<UNKsymbol>"] = 2
id_to_word = {value:key for key, value in word_to_id.items()}

#%% Prepare data
if numSamples==9000:
    x_train = x_train_orig[:numSamples]
    x_test  = x_train_orig[numSamples:(numSamples*2)]
    x_val   = x_train_orig[numSamples*2:(numSamples*2 + int(numSamples_val))]
    # total_words = len(set(np.concatenate(x_train).ravel()))
    x_train_labels = y_train[:numSamples]
    x_test_labels  = y_train[numSamples:(numSamples*2)]
    x_val_labels  = y_train[numSamples*2:(numSamples*2 + int(numSamples_val))]

if numSamples==3000:
    x_train = x_train_orig[:numSamples]
    x_test  = x_test_orig[:numSamples]
    x_val   = x_test_orig[numSamples:(numSamples + int(numSamples_val))]
    # total_words = len(set(np.concatenate(x_train).ravel()))
    x_train_labels = y_train[:numSamples]
    x_test_labels  = y_test[:numSamples]
    x_val_labels   = y_test[numSamples:(numSamples + int(numSamples_val))]

#train
x_train_text = []
for i, line in enumerate(x_train):
    decoded = [id_to_word.get(i) for i in line]
    decoded_review = ' '.join(decoded) # join word using ' ' between them
    x_train_text.append(decoded_review)

# Tokenization - Create vocabulary that fits the samples in the sub-dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train_text)
reverse_tokenizer = dict(map(reversed, tokenizer.word_index.items()))

x_train_tokens = []
for line in x_train_text:
    current_line_text = line.split(' ')
    current_line_tokens = tokenizer.texts_to_sequences(current_line_text)
    x_train_tokens.append(np.concatenate(current_line_tokens))

#test
x_test_text = []
for i, line in enumerate(x_test):
    decoded = [id_to_word.get(i) for i in line]
    decoded_review = ' '.join(decoded) # join word using ' ' between them
    x_test_text.append(decoded_review)

x_test_tokens = []
for line in x_test_text:
    current_line_text = line.split(' ')
    current_line_tokens = tokenizer.texts_to_sequences(current_line_text)
    x_test_tokens.append(np.concatenate(current_line_tokens))

#val
x_val_text = []
for i, line in enumerate(x_val):
    decoded = [id_to_word.get(i) for i in line]
    decoded_review = ' '.join(decoded)  # join word using ' ' between them
    x_val_text.append(decoded_review)

x_val_tokens = []
for line in x_val_text:
    current_line_text = line.split(' ')
    current_line_tokens = tokenizer.texts_to_sequences(current_line_text)
    x_val_tokens.append(np.concatenate(current_line_tokens))


#%%
total_words = len(tokenizer.word_index) + 1 # +1 for pad symbol (0). startsymbol is already in vcb.

max_sequence_len1 = max([len(s) for s in x_train]) ### max length in data
max_sequence_len2 = max([len(s) for s in x_test]) ### max length in data
max_sequence_len = max(max_sequence_len1, max_sequence_len2) ### max length in data

#%%
class KerasBatchGenerator(object):

    def __init__(self, x_train, x_train_labels, max_sequence_len, total_words, batch_size):
        # num_steps – the number of words fed into the input layer of the network. This is the set of words that the model will learn
        # from to predict the words coming after.
        # skip_steps is the number of words we want to skip over between training samples within each batch

        self.x_train = x_train
        self.x_train_labels = x_train_labels
        self.max_sequence_len = max_sequence_len
        self.total_words = total_words
        self.batch_size = batch_size
        self.current_line = 0
        self.current_idx = 2

    def generate(self):
        x = np.zeros((self.batch_size, self.max_sequence_len))
        x_labels = np.zeros((self.batch_size, self.max_sequence_len))
        y = np.zeros((self.batch_size, self.total_words))
        while True:
            for i in range(self.batch_size):
                if self.current_idx >= len(self.x_train[self.current_line]):
                    # reset the index
                    self.current_line += 1
                    self.current_idx = 2
                if self.current_line >= len(self.x_train):
                    # reset the index
                    self.current_line = 0

                line = self.x_train[self.current_line]
                label = self.x_train_labels[self.current_line]

                input_sequence = line[:self.current_idx]
                label_sequence = np.full_like(input_sequence, label)

                input_sequence_pad = np.array(pad_sequences([input_sequence], maxlen=self.max_sequence_len, padding=pad_type))
                label_sequence_pad = np.array(pad_sequences([label_sequence], maxlen=self.max_sequence_len, padding=pad_type))

                x[i, :] = input_sequence_pad
                x_labels[i, :] = label_sequence_pad

                temp_y = line[self.current_idx]
                # convert all of temp_y into a one hot representation
                y[i, :] = keras.utils.to_categorical(temp_y, num_classes=self.total_words)
                self.current_idx += 1

            yield [x, x_labels], y

#%%
def load_glove_embeddings(fp, embedding_dim, include_empty_char=True):
    """
    Loads pre-trained word embeddings (GloVe embeddings)
        Inputs: - fp: filepath of pre-trained glove embeddings
                - embedding_dim: dimension of each vector embedding
                - generate_matrix: whether to generate an embedding matrix
        Outputs:
                - word2coefs: Dictionary. Word to its corresponding coefficients
                - word2index: Dictionary. Word to word-index
                - embedding_matrix: Embedding matrix for Keras Embedding layer
    """
    # First, build the "word2coefs" and "word2index"
    word2coefs = {} # word to its corresponding coefficients
    word2index = {} # word to word-index
    with open(fp, encoding="utf8") as f:
        for idx, line in enumerate(f):
            try:
                data = [x.strip().lower() for x in line.split()]
                word = data[0]
                coefs = np.asarray(data[1:embedding_dim+1], dtype='float32')
                word2coefs[word] = coefs
                if word not in word2index:
                    word2index[word] = len(word2index)
            except Exception as e:
                print('Exception occurred in `load_glove_embeddings`:', e)
                continue
        # End of for loop.
    # End of with open
    if include_empty_char:
        word2index[''] = len(word2index)
    # Second, build the "embedding_matrix"
    # Words not found in embedding index will be all-zeros. Hence, the "+1".
    vocab_size = len(word2coefs)+1 if include_empty_char else len(word2coefs)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word2index.items():
        embedding_vec = word2coefs.get(word)
        if embedding_vec is not None and embedding_vec.shape[0]==embedding_dim:
            embedding_matrix[idx] = np.asarray(embedding_vec)
    # return word2coefs, word2index, embedding_matrix
    return word2index, np.asarray(embedding_matrix)

#%% Create generators
train_data_generator = KerasBatchGenerator(x_train_tokens, x_train_labels,  max_sequence_len-1, total_words, batch_size)
test_data_generator = KerasBatchGenerator(x_test_tokens, x_test_labels,  max_sequence_len-1, total_words, batch_size)
val_data_generator = KerasBatchGenerator(x_val_tokens, x_val_labels,  max_sequence_len-1, total_words, batch_size)

#%% callbacks

path_log = os.path.join(os.path.dirname(os.path.realpath(__file__)), log_dirname)

if os.path.exists(path_log):
    shutil.rmtree(path_log)
    os.makedirs(path_log)
else:
    os.makedirs(path_log)

currentDT = datetime.datetime.now()
currentDT_str = str(currentDT)
currentDT_str = currentDT_str.replace('-', '_')
currentDT_str = currentDT_str.replace(':', '_')
currentDT_str = currentDT_str.replace(' ', '_')

log_file_name = currentDT_str + '_training.log'
log_file_path = os.path.join(path_log, log_file_name)
csv_logger = CSVLogger(log_file_path, append=False)

trained_models_path = path_log
# model_names     = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_names = os.path.join(trained_models_path, '{epoch:02d}.hdf5')
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=2, save_best_only=False)

tensorboard = TensorBoard(log_dir="log_regressor_tb/{}".format(log_dirname))
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

callbacks = [model_checkpoint, csv_logger, tensorboard]

#%% Build model functions
# word2index start from 0 (inclusive)
word2index, embedding_matrix = load_glove_embeddings('glove.6B\\glove.6B.' + str(embedding_dim) + 'd.txt', embedding_dim=embedding_dim)

x_train_tokens = []
for line in x_train_text:
    current_line_text = text_to_word_sequence(line)
    # print(current_line_text)
    current_line_tokens = []
    for word in current_line_text:
        word = word.replace("'", "")
        if word == 'startsymbol':
            token = 1
        else:
            token = word2index.get(word)
            if token == None:
                token = 2
            else:
                token = token + 3 #start from 3 instead of 0 to have pad = 0 , startsymbol=1, unksymbol=2
        current_line_tokens.append(token)
    # print(current_line_tokens)
    x_train_tokens.append(current_line_tokens)

x_test_tokens = []
for line in x_test_text:
    current_line_text = text_to_word_sequence(line)
    # print(current_line_text)
    current_line_tokens = []
    for word in current_line_text:
        word = word.replace("'", "")
        if word == 'startsymbol':
            token = 1
        else:
            token = word2index.get(word)
            if token == None:
                token = 2
            else:
                token = token + 3 #start from 3 instead of 0 to have pad = 0 , startsymbol=1, unksymbol=2
        current_line_tokens.append(token)
    # print(current_line_tokens)
    x_test_tokens.append(current_line_tokens)

def build_model(input_len, total_words, embedding_dim, batch_size):
    input_seq = Input(shape=[input_len])

   
    # use pre-trained embedding
    Emb_x = Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      input_length=max_sequence_len-1,
                      weights=[embedding_matrix],
                      trainable=False)(input_seq)

    W_Emb_x = Dense(embedding_dim, activation='softmax')(Emb_x)

    input_label = Input(shape=[input_len])
    Emb_label = Embedding(total_labels, embedding_dim, input_length=input_len)(input_label)
    W_Emb_label = Dense(embedding_dim, activation='softmax')(Emb_label)

    lstm_in = Add()([W_Emb_x, W_Emb_label])
    lstm_out = LSTM(rnn_cell_size)(lstm_in)
	
    net_out = Dense(total_words, activation='softmax')(lstm_out)
    final_model = Model(inputs=[input_seq, input_label], outputs=[net_out], name="Language Model")
    print(final_model.summary())
    return final_model


def perplexity(y_true, y_pred):
    cross_entropy = keras.backend.mean(keras.backend.categorical_crossentropy(y_true, y_pred))
    perplexity = keras.backend.exp(cross_entropy)
    return perplexity


# A convergence graph - a graph with epochs as x-axis, and accuracy as y-axis.
# plot the train and test graphs for each tmodel in the same plot
def plotConvergence(history):
    plt.plot(history.history['perplexity'])
    plt.plot(history.history['val_perplexity'])
    plt.title('model perplexity')
    plt.ylabel('perplexity')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def generate_text(in_text, sentiment, next_words, max_sequence_len):
    out_text = in_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([out_text])[0]
        label_list = np.full_like(token_list, sentiment)

        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding=pad_type)
        label_list = pad_sequences([label_list], maxlen=max_sequence_len - 1, padding=pad_type)

        predicted = np.argmax(model.predict([token_list, label_list]))

        output_word = [reverse_tokenizer.get(predicted)]
        # seed_text = seed_text[0] + " " + output_word[0]
        out_text = out_text + output_word
        # print(seed_text)
    return out_text

#%% Build model
input_len = max_sequence_len-1
model = build_model(input_len, total_words, embedding_dim, batch_size)


#%%
opt = keras.optimizers.adam(lr=learning_rate)

# Compile model (required to make predictions)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy', perplexity])

print("model metrics: ", model.metrics_names)

n_samples_in_data = sum([len(s)-1 for s in x_train_tokens]) ### max length in minibatch
steps_per_epoch = n_samples_in_data // batch_size

n_samples_in_data_val = sum([len(s) - 1 for s in x_val_tokens])  ### max length in minibatch
steps_per_epoch_val = n_samples_in_data // batch_size

# Train
history = model.fit_generator(train_data_generator.generate(), epochs=epochs, steps_per_epoch=steps_per_epoch,
                    validation_data=test_data_generator.generate(), validation_steps=steps_per_epoch,
                    callbacks=callbacks, verbose=1)

plotConvergence(history)

#%% Run trained model
# input text
in_text = 'startsymbol'

in_text = keras.preprocessing.text.text_to_word_sequence(in_text, lower=True, split=" ")

num_word_to_predict = 20  # number of words to predict
sentiment = 1  # desired sentiment: 1 or 2

# run prediction
out_text = generate_text(in_text, sentiment, num_word_to_predict, max_sequence_len)
print(out_text)
