# import libraies
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline ## eski veriyi hafizasında tutar yeni gelen veriyi rakama çevirir öğretir
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# imprt data
model=load_model("Traffic_model.h5")

def process_image(img):
    img=img.resize((30,30))  # Boyutunu 30*30 px yaptik
    img=np.array(img)
    img=img/255.0  #Normalize ettik
    img=np.expand_dims(img,axis=0)  # dizinin boyutunu genişletiyor
    return img

st.title("Almanyadaki trafik isaretlerini taniyan model :no_bicycles:")
st.write("Resim sec model hangi tarfik isareti oldugunu tahmin etsin")

file=st.file_uploader("Bir resim sec",type=["jpeg","jpg","png"])

# Kameradan fotoğraf yükleme
file = st.camera_input("Yada Kameranizi kullanarak bir fotograf cek:")

if file:
    st.image(file, caption="Çekilen Fotoğraf", use_column_width=True)


if file is not None:
    img=Image.open(file)
    st.image(img,caption="yuklenen resim")
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)

class_names= {  0:'Speed limit (20km/h)',
                1:'Speed limit (30km/h)', 
                2:'Speed limit (50km/h)', 
                3:'Speed limit (60km/h)', 
                4:'Speed limit (70km/h)', 
                5:'Speed limit (80km/h)', 
                6:'End of speed limit (80km/h)', 
                7:'Speed limit (100km/h)', 
                8:'Speed limit (120km/h)', 
                9:'No passing', 
                10:'No passing veh over 3.5 tons', 
                11:'Right-of-way at intersection', 
                12:'Priority road', 
                13:'Yield', 
                14:'Stop', 
                15:'No vehicles', 
                16:'Veh > 3.5 tons prohibited', 
                17:'No entry', 
                18:'General caution', 
                19:'Dangerous curve left', 
                20:'Dangerous curve right', 
                21:'Double curve', 
                22:'Bumpy road', 
                23:'Slippery road', 
                24:'Road narrows on the right', 
                25:'Road work', 
                26:'Traffic signals', 
                27:'Pedestrians', 
                28:'Children crossing', 
                29:'Bicycles crossing', 
                30:'Beware of ice/snow',
                31:'Wild animals crossing', 
                32:'End speed + passing limits', 
                33:'Turn right ahead', 
                34:'Turn left ahead', 
                35:'Ahead only', 
                36:'Go straight or right', 
                37:'Go straight or left', 
                38:'Keep right', 
                39:'Keep left', 
                40:'Roundabout mandatory', 
                41:'End of no passing', 
                42:'End no passing veh > 3.5 tons' }

st.write(class_names[predicted_class])
