def save_weights(epoch, model,model_dir):
    if not os.path.exists(model_dir):  
        os.makedirs(model_dir) # create directory if it does not exist
    model.save_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch))) # save weights

def load_weights(epoch, model,model_dir):
    model.load_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch))) # load weights