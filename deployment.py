import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import backend as K
from glob import glob
import numpy as np
import os
import streamlit as st
from PIL import Image

#Read data 

import pandas as pd



def rle_decode(mask_rle, shape):
    '''mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction




weight_for_0_large = 1
weight_for_1_large = 1

class_weights_large = {0:weight_for_0_large, 1:weight_for_1_large}

def custom_binary_loss(y_true, y_pred):
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)
  y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
  term_0 = 1.4 * weight_for_0_large*((1 - y_true) * K.log(1 - y_pred + K.epsilon()))  # Cancels out when target is 1 
  term_1 = 0.55 * weight_for_1_large*(y_true * K.log(y_pred + K.epsilon())) # Cancels out when target is 0

  return -K.mean(term_0 + term_1, axis=-1)


best_model = tf.keras.models.load_model('best_model_AML.h5', custom_objects={'custom_binary_loss': custom_binary_loss})


img_path = df_train['path'][1]
img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
img = (img - img.min())/(img.max() - img.min())*255.0 
img = cv2.resize(img, (256, 256))
img = np.expand_dims(img, axis=-1)
img = img.astype(np.float32) / 255.
pred = best_model.predict(np.array([img]))




def color_class(pred):
  classes = {'large bowel':0.3, 'small bowel':0.5, 'stomach':0.9}
  threshold = 0.95
  t = 0.94999
  pixels_to_change_large = np.where(pred[0][0] > threshold)
  pixels_to_change1_large = np.where(pred[0][0] < t)

  pixels_to_change_small = np.where(pred[1][0] > threshold)
  pixels_to_change1_small = np.where(pred[1][0] < t)

  pixels_to_change_stomach = np.where(pred[2][0] > threshold)
  pixels_to_change1_stomach = np.where(pred[2][0] < t)
  # Modify identified pixels
  pred[0][0][pixels_to_change_large] = classes['large bowel']
  pred[0][0][pixels_to_change1_large] = 0
  pred[1][0][pixels_to_change_small] = classes['small bowel']
  pred[1][0][pixels_to_change1_small] = 0
  pred[2][0][pixels_to_change_stomach] = classes['stomach']
  pred[2][0][pixels_to_change1_stomach] = 0

  tot_mask = pred[0][0] + pred[1][0] + pred[2][0]

  return tot_mask

mask_pred = color_class(pred)

plt.figure(figsize=(5*5, 7))
plt.imshow(img, cmap='bone', alpha=1)
plt.imshow(mask_pred, alpha=0.45, cmap='hot')
plt.axis('off')


def main():
   
   
   st.title('Optimización del tratamiento de cáncer GI')
   st.markdown('Elige una imagen de validación para probar el modelo.')

   option = st.selectbox('Imagen:',('81', '82', '83'))
   
   path = '2slice_00'+ option +'_266_266_1.50_1.50.png'


   st.write('Imagen seleccionada:', option)


   


   img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
   img = (img - img.min())/(img.max() - img.min())*255.0 
   img = cv2.resize(img, (256, 256))
   img = np.expand_dims(img, axis=-1)
   img = img.astype(np.float32) / 255.
   fig, ax = plt.subplots()
   ax.imshow(img)


   pred = best_model.predict(np.array([img]))
   mask_pred = color_class(pred)



   fig1 = plt.figure(figsize = (3,3))
   plt.subplot(2, 2, 1)
   plt.imshow(img)
   plt.subplot(2, 2, 2)
   plt.imshow(img, cmap='bone', alpha=1)
   plt.imshow(mask_pred, alpha=0.45, cmap='hot')

   st.write('Imagen seleccionada')
   st.pyplot(fig1)


   #fig1, ax1 = plt.subplots()
   #ax1.imshow(img, cmap='bone', alpha=1)
   #ax1.imshow(mask_pred, alpha=0.45, cmap='hot')
   #st.write('Imagen con predicción:')
   #st.pyplot(fig1)




if __name__ == '__main__':
    main()