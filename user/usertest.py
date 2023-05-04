import numpy as np
from PIL import Image
from keras import models
import os
import shutil

filepath = './images/'

classes = ['aguacate', 'kiwi', 'limon', 'naranja', 'piña']
model = models.load_model('../model/model.h5')

def get_class(filename):
    return filename.split('_')[0]

index = 0

while True:
    if len(os.listdir(filepath)) != 0:
        try:
            filename = os.listdir(filepath)[index]
            path = filepath + filename

            img = Image.open(path)
            img = img.resize((348, 348))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = classes[np.argmax(model(img_array), axis=1)[0]]
            expected = get_class(filename)

            if prediction == expected:
                print(f'Predicción correcta: {prediction}')          
                shutil.move(path, f'./clasified/{filename}')
            else:
                print(f'Fallo en la predicción: predicción {prediction} - esperado {expected}')
                shutil.move(path, f'./wrong/{filename}')
            
            index = 0

        except Exception as e:
            print(f'{e=}')
            index += 1