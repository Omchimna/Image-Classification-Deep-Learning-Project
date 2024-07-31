import cv2
import streamlit as st
import numpy as np
# from PIL import Image

def main():
  st.title("Image Classification Of Fashion Items")
  st.text("This Works")  
  uploaded_file = st.file_uploader("Upload Image")
  image = cv2.imread(uploaded_file)   
  # image=image[:, ::-1]
  # BGR -> RGB  
  #type(image.shape)
  # img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
  small_img=cv2.resize(image, (28,28))
  img=cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY,)
  # img=np.abs(256-img)
  fig1=plt.figure()
  plt.subplot(1,2,1)
  plt.imshow(img)
  # plot_image(i, predictions[i], test_labels, test_images)
  # plt.subplot(1,2,2)
  # plot_value_array(i, predictions[i],  test_labels)
  plt.show()
  st.fig(fig1)

if __name__ == '__main__':
  main()
