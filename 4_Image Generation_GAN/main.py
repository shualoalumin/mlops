import tensorflow as tf
import PIL
import imageio

# TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# PIL (Pillow) version
print(f"PIL version: {PIL.__version__}")

# imageio version
print(f"imageio version: {imageio.__version__}")

def make_generator_model():
    from tensorflow.keras import layers
    
    model = tf.keras.Sequential()
    
    # [First] Dense layer (units=7*7*256, no bias, input shape=100)
    # BatchNormalization -> LeakyReLU
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # [Second] Reshape layer
    model.add(layers.Reshape((7, 7, 256)))
    
    # [Third] Conv2DTranspose (kernel_size=5, strides=1, padding='same', no bias)
    # BatchNormalization -> LeakyReLU
    model.add(layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 1), 
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # [Fourth] Conv2DTranspose (kernel_size=5, strides=2, padding='same', no bias)
    # BatchNormalization -> LeakyReLU
    model.add(layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), 
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # [Fifth] Conv2DTranspose (output layer: 1 channel, tanh activation)
    model.add(layers.Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), 
                                     padding='same', use_bias=False, 
                                     activation='tanh'))
    
    return model

