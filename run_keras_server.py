# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.applications import imagenet_utils
import numpy as np
import flask
import io
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 196

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False
model = None

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = tf.keras.models.load_model('my_checkpoint')

def sentence_generation(model, tokenizer, current_word,n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    sentence = list()

    # n번 반복
    for _ in range(n):
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=MAX_LEN-1, padding='pre')

        # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for word, index in tokenizer.word_index.items(): 
            # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
            if index == result:
                break

        # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        current_word = current_word + ' '  + word

        # 예측 단어를 문장에 저장
        sentence.append(word) 

    return list(set(sentence))


@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
    data = {"success": False, "predictions":[]}
    # ensure an image was properly uploaded to our endpoint 
    if flask.request.method == "POST":
        params = flask.request.get_json()
        print(params["text"])
        text = params["text"]
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        r = sentence_generation(model, tokenizer, text, 10)
          
        data["predictions"] = r

		# indicate that the request was a success
        data["success"] = True

	# return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
    load_model()
    app.run()
    print('hello')