import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import streamlit as st
from streamlit_drawable_canvas import st_canvas


st.title("Digit Recognizer")
st.write("draw the number")
canvas_result = st_canvas(fill_color='#000000', stroke_width=20, stroke_color='#FFFFFF',background_color='#000000',
    width=150,height=150,drawing_mode="freedraw",key='canvas')

if canvas_result.image_data is not None:
    img= cv2.resize(canvas_result.image_data.astype('uint8'),(28, 28))
    rescaled = cv2.resize(img,(200,200 ), interpolation=cv2.INTER_NEAREST)
    st.write('rescaled input')
    st.image(rescaled)

model=keras.models.load_model("digit.hdf5")

if st.button('Predict'):
    sample = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pr = model.predict(sample.reshape(1, 28, 28))
    st.write(f'Output is: {np.argmax(pr[0])}')
    st.bar_chart(pr[0])