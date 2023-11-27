import streamlit as st
import pickle
import mahotas

rf = pickle.load(open("./Machine_Learning/rf.pkl", "rb"))
gbm = pickle.load(open("./Machine_Learning/gbm.pkl", "rb"))
xgb = pickle.load(open("./Machine_Learning/xgb.pkl", "rb"))
svm = pickle.load(open("./Machine_Learning/svm.pkl", "rb"))


st.title("Brain Tumor Detection")
# st.markdown("Here we are using zernike moments extracted from the tumor images")

# doing a select box to ask the user to choose a model
models = [
    (rf, "1. Random Forest"),
    (gbm, "2. Gradient Boosting"),
    (xgb, "3. Xtreme Gradient Boosting (XgBoost)"),
    (svm, "4. Support Vector Machine"),
]
model_names = list(map(lambda t: t[1], models))

choosen_model = st.selectbox(
    "Choose your preferred prediction model. Some models give better prediction, So try all :)",
    options=model_names,
)

print(choosen_model)
model = models[model_names.index(choosen_model)][0]

st.subheader("Upload the MRI image of brain.")

try:
    # deg = st.text_input('', 0,100)
    deg = st.file_uploader("Upload an image", [".jpg", ".png"])

    # print(deg.name)
    # reading image into mahatos
    img = mahotas.imread(deg, as_grey=True)
    actual_img = mahotas.features.zernike_moments(img, radius=32, degree=32).reshape(
        (1, 289)
    )
    # print(actual_img.__len__())
    print(actual_img.shape)

    st.subheader("Predicted Result")
    result = None

    st.subheader(f"Based on the {choosen_model[3:]}, your results are as follows")
    if int(model.predict(actual_img)):
        result = "It seems your results are POSITIVE and you have BRAIN TUMOR. Allah will help you."
        # result = "It seems you have BRAIN TUMOR"
    else:
        result = "You are safe. Jesus loves you. Always"

    st.text(result)

except Exception as e:
    st.text("Something is wrong. Try uploading the MRI image of brain.")
