import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import numpy as np
from PIL import Image
import os
from skimage import measure
import pandas as pd

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

@tf.function
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def get_mask_as_image(mask):
    mask = mask.flatten().reshape(768,768).tolist()
    threshold = 0.5
    res = []
    for i in range(768):
        res.append([])
        for j in range(768):
            if (mask[i][j]>threshold):
                res[i].append(1)
               # print('yes')
            else:
                res[i].append(0)
    #print(res)
    #mask_image = Image.fromarray(np.array(res).astype('uint8')*255)
    #mask_image = mask_image.convert('RGB')
    return np.array(res).astype('uint8')

def _parse_function_1(image_path):
    image_string = tf.io.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    #image_decoded = tf.image.resize(image_decoded, [128, 128])
    #image_decoded = tf.image.resize(image_decoded, [128, 128])
    image = tf.cast(image_decoded, tf.float32)/255.0
    #print(image)
    return np.expand_dims(image, axis=0)

def ship_box_to_rle(box,mask):
    startx = box[1]
    endx = box[3]
    stary = box[0]
    endy = box[2]
    res = ''
    cnt = 0
    pos = startx*len(mask)+stary+1
    endpos = endx*len(mask)+endy+1
    curpos = startx*len(mask)+stary+1
    while curpos<=endpos:
        x = curpos//len(mask)
        y = curpos%len(mask)-1
        if (mask[x][y]==1):
            
            cnt+=1
            if (cnt==1):
                pos = curpos
        else:
            if (cnt>0):
                if res=='':
                    res = str(pos)+' '+str(cnt)
                else:
                    res = res + ' '+str(pos)+' '+str(cnt)
            cnt = 0  
        curpos+=1
    if cnt>0:
        res = res + ' '+str(pos)+' '+str(cnt)
    return res
        
def clusterize_objects(segmentation_mask):
    # Label connected components in the segmentation mask
    labeled_mask, num_objects = measure.label(segmentation_mask, connectivity=2, return_num=True)

    # Create an array to store information about each labeled object
    object_info = []

    # Iterate through each labeled object
    for obj_id in range(1, num_objects + 1):  # Start from 1 since 0 is background
        object_pixels = np.where(labeled_mask == obj_id)
        #print('HEllo')
        #print(object_pixels)
        object_bbox = (
            min(object_pixels[1]),
            min(object_pixels[0]),
            max(object_pixels[1]),
            max(object_pixels[0])
        )
        object_info.append({
            'id': obj_id,
            'bbox': object_bbox,
            'pixels': object_pixels
        })

    return object_info

def create_rle_from_mask(mask):
    clusterize_objs = list(clusterize_objects(np.array(mask)))
    print(clusterize_objs)
    plt.imshow(Image.fromarray(np.array(mask)*255))
    plt.show()
    res = []
    for i in range(len (clusterize_objs)):
        res.append(ship_box_to_rle(np.array(clusterize_objs[i]['bbox']).tolist(),mask))
    return res


model = tf.keras.models.load_model('./model_dice_fix_bugs.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss})
directory = './data/test_v2'
ind = 0

image_list = []
rle_list = []

for filename in os.listdir(directory):
    ind+=1
    f = os.path.join(directory, filename)
    image = _parse_function_1(f)
    #print(model.output_shape)
    #print(model.summary())
    #print(image.shape)
    img_str = f
    mask =  model.predict([_parse_function_1(f)])[0]
    #mask2 = model2.predict([_parse_function_2(f)])[0]
    mask = get_mask_as_image(mask)
    rle_tmp = create_rle_from_mask(mask)
    print(rle_tmp)
    if (len(rle_tmp) > 20 or len(rle_tmp)==0):
        image_list.append(filename)
        rle_list.append(None)
    else:
        for i in rle_tmp:
            image_list.append(filename)
            rle_list.append(i)
    
df = pd.DataFrame({'ImageId':image_list,'EncodedPixels':rle_list})
df.to_csv('./res.csv')