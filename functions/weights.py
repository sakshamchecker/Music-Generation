from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
import os

def save_weights(epoch, model,model_dir):
    if not os.path.exists(model_dir):  
        os.makedirs(model_dir) # create directory if it does not exist
    model.save_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch))) # save weights

def load_weights(epoch, model,model_dir):
    model.load_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch))) # load weights