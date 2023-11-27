import streamlit as st
import pickle
import mahotas

model = pickle.load(open('./Machine_Learning/xgb.pkl','rb'))

st.title("Brain Tumor Detection")
st.markdown("Here we are using zernike moments extracted from the tumor images")

st.subheader("Enter the Zernike moment")
# deg = st.text_input('', 0,100)
deg = st.file_uploader("Upload an image", [".jpg", '.png'])

# print(deg.name)
# reading image into mahatos
img = mahotas.imread(deg, as_grey=True)
actual_img = mahotas.features.zernike_moments(img, radius=32, degree=32)
# print(actual_img.__len__())

st.subheader("Predicted Revenue")
st.code(float(model.predict([[actual_img]])))
