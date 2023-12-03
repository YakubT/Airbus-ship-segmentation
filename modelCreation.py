import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,  BatchNormalization, Activation, Dropout, Conv2DTranspose
import matplotlib.pyplot as plt
import os
import keras.backend as K
from keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import binary_crossentropy
import numpy as np

tf.config.run_functions_eagerly(True)

print(tf.config.list_physical_devices('GPU'))

tf.random.set_seed(777)

# Set random seed for NumPy
np.random.seed(777)

def extract_channel(mask, channel_index):
    # Assuming 'mask' is a tensor with shape (height, width, channels)
    extracted_channel = mask[:, :, channel_index+1]

    # Add an extra dimension to make it (height, width, 1)
    extracted_channel = tf.expand_dims(extracted_channel, axis=-1)

    return extracted_channel


def _parse_function(image_path, mask_path):
    image_string = tf.io.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)/255.0
    mask_string = tf.io.read_file(mask_path)
    mask_decoded = tf.image.decode_jpeg(mask_string, channels=3)
    mask_decoded = tf.image.rgb_to_grayscale(mask_decoded)
    mask = tf.cast(mask_decoded, tf.float32)/255.0
    mask = np.round(mask.numpy())
    return image, mask




def unet_model(input_shape=(768, 768, 3)):
    # Input layer
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
# Bottleneck
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    up1 = UpSampling2D((2, 2))(conv3)
    concat1 = concatenate([conv2, up1], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = UpSampling2D((2, 2))(conv4)
    concat2 = concatenate([conv1, up2], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)  # Assuming binary segmentation

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model




threshold = 0.5



def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(K.flatten(y_true),'float32')
    y_pred_f = tf.cast(K.flatten(y_pred), 'float32')
    print(y_true_f.numpy())
    y_pred_f = tf.where(y_pred_f < threshold, 0.0, y_pred_f )
    y_pred_f = tf.where(y_pred_f >= threshold, 1.0, y_pred_f )
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
    return dice

def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)

def thresholded_binary_crossentropy(y_true, y_pred):
    y_pred = tf.where(y_pred < threshold, 0.0, 1.0)
    return K.binary_crossentropy(y_true, y_pred)

path = '.'

def weighted_binary_crossentropy(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights = (y_true_f * 255.) + 1.
    bce = K.binary_crossentropy(y_true_f, y_pred_f)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

def getDs(path_img,path_masks,cnt):
    
    image_paths = []
    mask_paths = []
    for i in range(1,cnt+1):
        image_paths.append(path_img+'/image'+str(i)+'.jpg')
        mask_paths.append(path_masks+'/mask'+str(i)+'.jpg')
    
    # Create a dataset from slices
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Map the load_and_preprocess_image function to load and preprocess each image and mask
    dataset = dataset.map(lambda x, y: tf.py_function(_parse_function, [x, y], [tf.float32, tf.float32]))
    
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(batch_size=2).prefetch(buffer_size=tf.data.AUTOTUNE) 
    return dataset

# Instantiate the model
model = unet_model()

optimizer = keras.optimizers.Adam(lr=0.001)

# Compile the model
model.compile(optimizer='adam',  loss = dice_coef_loss)

# Display the model summary
model.summary()

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = getDs(path+'/processed_data_train/train/images',path+'/processed_data_train/train/masks',1400)

#_parse_function(path+'/processed_data_train/train/images/image2.jpg',path+'/processed_data_train/train/masks/mask2.jpg')

validation_dataset = getDs(path+'/processed_data_train/validation/images',path+'/processed_data_train/validation/masks',360)
#train
print(len(train_dataset))

history = model.fit(train_dataset,validation_data = (validation_dataset),epochs=5)
model.save('./model_dice_fix_bugs.hdf5')