import cv2
import streamlit as st
import numpy as np
from PIL import Image, ImageOps 
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./final_model.keras')
	return model

def final_img(file):
	# image = cv2.imread(file)
	img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1) 
	img_1 = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY,)
	img_2 = cv2.resize(img_1, (28,28), )
	img_3 = (np.expand_dims(img_2,0))
	return img_3

def predict_class(image, model):
	pre_img = tf.convert_to_tensor(image)
	prediction = model.predict(pre_img)

	return prediction

def main():
	st.title("Image Classification Of Fashion Items")
	st.text("This Works")  
	uploaded_file = st.file_uploader("Upload Image" ,type=['jpg', 'png', 'jpeg'])
	if uploaded_file is not None:
		final = final_img(uploaded_file)
		model = load_model()
		result = predict_class(final,model)
		probabilities = result[0]
		# img=np.abs(256-img)

		class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
		ax=plt.figure(figsize=(20,5))
		plt.imshow(final[0])
		plt.subplot(2,1,1)
		
		plt.axis('off')
		# st.image(ax1)
		plt.bar(class_names, probabilities)
		plt.xticks(range(10), class_names, rotation=45)
		plt.ylabel('Probability')
		plt.subplot(2,1,2)
		plt.axis('off')
		st.pyplot(ax)

if __name__ == '__main__':
	main()
