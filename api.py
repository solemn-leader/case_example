import flask
from flask import request
import os
import onnxruntime as rt
from transformers import DistilBertTokenizer
import numpy as np


port = int(os.environ.get('PORT', 5000))

app = flask.Flask(__name__)
app.config['DEBUG'] = True

tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased'
)
model = rt.InferenceSession(
    'models/model-quantized.onnx',
    providers=['CPUExecutionProvider']
)
classes = [
    'Government News', 'Middle-east', 'News', 
    'US_News', 'left-news', 'politics'
]


def predict(text):
    model_inputs = tokenizer(
        text, truncation=True, padding=True
    )
    model_inputs = {
        k: np.array([v]) for k, v in model_inputs.items()
    }
    outp = model.run(
        None, model_inputs
    )
    class_ind = outp[0][0].argmax()
    return classes[class_ind]


@app.route('/classify', methods=['GET'])
def classify_news():
    text = str(request.query_string).replace('%20', ' ')[2: -1]
    class_name = predict(text)
    return '<h1>' + class_name + '</h1>'


app.run(
    host='0.0.0.0',
    port=port
)
