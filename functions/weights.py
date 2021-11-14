def save_weights(epoch, model,model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch)))

def load_weights(epoch, model,model_dir):
    model.load_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch)))