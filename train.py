import os
import json
import argparse

import numpy as np

from functions import batch_read, set_model 
from functions.weights import save_weights, load_weights
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

data = './data/input.txt'
model_dir='./models'
BATCH_SIZE = 207
SEQ_LENGTH = 64


file1 = open(data,"r")
data=file1.read()

def train(text, epochs, save_freq):


    char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }
    print("Number of unique characters: " + str(len(char_to_idx))) #95
    
    with open('./data/char_to_idx.json', 'w') as f:
        json.dump(char_to_idx, f)

    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)

    model = set_model.set_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32) 

    print("Length of text:" + str(T.size)) 

    steps_per_epoch = (len(text) / BATCH_SIZE - 1) / SEQ_LENGTH  


    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        
        losses, accs = [], []

        for i, (X, Y) in enumerate(batch_read.batch_read(T, vocab_size, BATCH_SIZE, SEQ_LENGTH)):
            

            loss, acc = model.train_on_batch(X, Y)
#             
            losses.append(loss)
            accs.append(acc)
        if (epoch + 1) % save_freq == 0:
            
            save_weights(epoch + 1, model, model_dir)
            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=5, help='checkpoint save frequency')
    args = parser.parse_args()
    train(data, args.epochs, args.freq)

