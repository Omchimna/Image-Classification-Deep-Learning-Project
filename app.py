import cv2
import streamlit as st
import numpy as np
from PIL import Image, ImageOps 
from matplotlib import pyplot as plt

def image_pre_pro(file):
  # image = cv2.imread(file)
  final_img = cv2.cvtColor(file.astype(np.uint8), cv2.COLOR_BGR2GRAY,)
  small_img = cv2.resize(final_img, (28,28), )
  return small_img

def main():
  st.title("Image Classification Of Fashion Items")
  st.text("This Works")  
  uploaded_file = st.file_uploader("Upload Image" ,type=['jpg', 'png', 'jpeg'])
  if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
  # image=image[:, ::-1]
  # BGR -> RGB  
  #type(image.shape)
  # img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    final=image_pre_pro(img)
  # img=np.abs(256-img)
  plt.figure()
  # plt.subplot(1,2,1)
  plt.imshow(final)
  # plot_image(i, predictions[i], test_labels, test_images)
  # plt.subplot(1,2,2)
  # plot_value_array(i, predictions[i],  test_labels)
  plt.show()
  # st.image(fig1)

if __name__ == '__main__':
  main()
