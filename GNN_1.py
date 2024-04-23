import streamlit as st
from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import h5py
import pickle

st.header("Leaf Disease predictor")

def main():
    file_uploaded = st.file_uploader("Choose the file", type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    pickle_in = open('classifier.pkl', 'rb')
    #classifier_model = tf.keras.models.load_model(r'')
    classifier_model = pickle.load(pickle_in)
    shape = ((224,224,3))
    model = tf.keras.Seuential(hub[hub.KerasLayer(classifier_model, input_shape = shape)])
    model = pickle.load(open(), input_shape = shape)
    test_image = image.resize((224, 224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    class_names = ['Blast', 'Tungro', 'Brownspot', 'Bacterialblight']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    images_class = class_names[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    return result

if __name__ == "__main__":
    main()
