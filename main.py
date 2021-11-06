import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf 
import pickle
############ Model Loading ######################
model = tf.keras.models.load_model("eff-model")
model1 = pickle.load(open("xgb_reg.pkl", 'rb'))
##################################################
st.markdown("<h1 style='text-align: center; color: black;'>Pawpularity Contest</h1>", unsafe_allow_html=True)
st.markdown('##')
# expander_bar = st.expander("About")
# expander_bar.markdown("""
# * **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
# * **Data source:** [CoinMarketCap](http://coinmarketcap.com).
# * **Credit:** Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf).
# """)
image = Image.open('Petfinder.jpg')
st.image(image, use_column_width=True)
st.markdown("""A picture is worth a thousand words. But did you know a picture can save a thousand lives? Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. You might expect pets with attractive photos to generate more interest and be adopted faster. But what makes a good picture? With the help of data science, you may be able to accurately determine a pet photo’s appeal and even suggest improvements to give these rescue animals a higher chance of loving homes.
     """)
st.markdown('If successful, your solution will be adapted into AI tools that will guide shelters and rescuers around the world to improve the appeal of their pet profiles, automatically enhancing photo quality and recommending composition improvements. ')
uploaded_file = st.file_uploader('Select an Image', ['png', 'jpg'])
w,h = 299,299
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    st.caption("""Your Image """)
    st.image(image1, use_column_width=True)
    img = np.expand_dims(np.array(image1.resize((w, h), resample=Image.BILINEAR))/255.,0)
    outputs2 = model.predict(img)
    st.write(outputs2.squeeze(0))
