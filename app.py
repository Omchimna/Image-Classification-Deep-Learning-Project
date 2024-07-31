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
	final_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY,)
	small_img = cv2.resize(final_img, (28,28), )
	return small_img

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
		fig,ax=plt.subplots(1,2)
		# ax.subplot(1,2,1)
		ax[0].imshow(final)
		# ax.subplot(1,2,2)
		ax[1].bar(class_names, probabilities)
		ax[1].xticks(range(10), class_names, rotation=45)
		ax[1].ylabel('Probability')
		st.pyplot(fig)

if __name__ == '__main__':
	main()
