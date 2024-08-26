import streamlit as st
import tensorflow as tf
import numpy as np

# Function to load the model and make a prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch.
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Side bar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease recognition"])

# Home page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home.jpg"
    st.image(image_path, use_column_width=True)
    st.write(
        "Plant diseases, caused by pathogens like fungi, bacteria, viruses, and nematodes, can severely affect crop yield and quality. "
        "Common diseases include powdery mildew, root rot, and leaf spot, each capable of devastating plant health. Early detection of these diseases is crucial to prevent widespread outbreaks, reduce economic losses, and ensure food security. "
        "Identifying diseases early allows for timely interventions, such as the application of fungicides, crop rotation, or the use of resistant plant varieties, ultimately safeguarding agricultural productivity and ecosystem health."
    )

elif app_mode == "About":
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
        This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
        A new directory containing 33 test images is created later for prediction purpose.
        #### Content
        1. train (70295 images)
        2. test (33 images)
        3. validation (17572 images)
    """)
    st.markdown("""
        #### About Model
        The model is a Convolutional Neural Network (CNN) designed for image classification, specifically for recognizing plant diseases. It features a sequential architecture with multiple Conv2D layers to extract features, MaxPooling2D layers for downsampling, Dropout layers for regularization, and Dense layers for final classification. The model has a total of 7,842,762 trainable parameters, with most concentrated in the Dense layers. During training over 10 epochs, the model demonstrated significant improvement in accuracy, rising from 39.79% to 98.03% on the training set, and from 82.14% to 94.36% on the validation set. Loss metrics also showed a substantial decrease, with training loss falling from 2.1383 to 0.0588 and validation loss from 0.5727 to 0.1907. The final evaluation showed a training accuracy of 97.58% and a validation accuracy of 94.36%, indicating strong performance and effective learning. The trained model has been saved for future use. Overall, the model exhibits robust performance in classifying plant diseases.
    """)
    image_path = "accuracy.png"
    st.image(image_path, use_column_width=True)

elif app_mode == "Disease recognition":
    st.header("Disease Recognition")
    
    # File uploader for the image
    test_image = st.file_uploader("Choose an image")
    
    if test_image:
        st.write("Processing your image, please wait...")
        
        with st.spinner('Analyzing the image...'):
            try:
                # Make prediction using the model
                result_index = model_prediction(test_image)
                
                # List of class names corresponding to the model's predictions
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                dic={'Corn_(maize)___Common_rust_':'Common Rust is a plant disease that affects maize (corn) and is caused by the fungus Puccinia sorghi. This disease is characterized by the appearance of reddish-brown pustules or lesions on the leaves, stems, and ears of maize plants.',
                'Peach___Bacterial_spot':'Bacterial Spot is a plant disease that affects peaches (and other stone fruits) caused by the bacterium Xanthomonas campestris pv. pruni. This disease can cause significant damage to peach trees and reduce fruit quality and yield.',
                }
                # Display the prediction result
                prediction_label = class_name[result_index]
                st.success(f"Model predicts: {prediction_label}")
                
                # Display the uploaded image
                st.image(test_image, use_column_width=True)
                
                # Check if the prediction is not a healthy label, then show the "Know More" button
                if "healthy" not in prediction_label.lower():
                    # Button to show more information
                    if st.button("Know More"):
                        if prediction_label in dic.keys():
                            st.info(f"{dic[prediction_label]}")


            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


