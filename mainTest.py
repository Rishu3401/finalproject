import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model(r'C:\Users\rishu\Downloads\final\final\BrainTumor10EpochsCategorical.h5')
image_path = r'C:\Users\rishu\Downloads\final\final\Training\yes\glioma\Tr-gl_0010.jpg'
image = cv2.imread(image_path)
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)
input_img = np.expand_dims(img, axis=0)
result = np.argmax(model.predict(input_img), axis=1)
print(result)
