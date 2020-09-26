# No Tensorflow info messages
import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from PIL import ImageOps

# Create class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load model
probability_model = keras.models.load_model('model')

# Predict on input image
img_path = input('Enter image path in form \'subdir/image.jpg\': ')
bgr = input('White or black background [W/b]? ')
bgr = bgr.upper()

img = Image.open(img_path).convert('L')

if bgr == 'W':
    img = ImageOps.invert(img)
elif bgr == 'B':
    pass
elif bgr == '':
    img = ImageOps.invert(img)
else:
    print('Could not read input\nExiting...')
    exit()

img = img.resize((28, 28))
img = np.array(img)
img = img / 255
img = (np.expand_dims(img, 0))
predictions_single = probability_model.predict(img)
print(class_names[np.argmax(predictions_single)])