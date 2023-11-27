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
st.markdown(
    """<p>
                <span style="color:red">*</span>
                Only upload MRI images. Not your face.
                </p>""",
    unsafe_allow_html=True,
)

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

    st.markdown(
        """
            <h4>
                <hr>
                Predicted Result
            </h4>
        """,
        unsafe_allow_html=True,
    )
    result = None

    st.markdown(
        f'<h5>Based on the "{choosen_model[3:]}", your results are as follows. </h5>',
        unsafe_allow_html=True,
    )
    if int(model.predict(actual_img)):
        st.markdown(
            '<h5 style="color:red">TUMOR DETECTED<br><p>It seems your results are POSITIVE and you have BRAIN TUMOR. Allah will help you.</p></h5>',
            unsafe_allow_html=True,
        )
        # result = "It seems you have BRAIN TUMOR"
    else:
        st.markdown(
            '<h5 style="color:green">TUMOR NOT DETECTED<br><p>You are SAFE. Jesus loves you. Always!!</p><h5>',
            unsafe_allow_html=True,
        )

    count = 0
    for model, model_name in models:
        count += int(model.predict(actual_img))
    st.markdown(f"<p>Average Probability of brain tumor based on above models is <span style=\"color:magenta; font-weight: Bold;\">{round(count/4, 2)*100}%</span></p>", unsafe_allow_html=True)


except Exception as e:
    st.text("Something is wrong. Try uploading the MRI image of brain.")
