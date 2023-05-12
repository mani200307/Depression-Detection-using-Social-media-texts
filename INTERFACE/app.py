from preprocess import *
from flask import Flask, render_template, request
import pickle

# Create flask app
app = Flask(__name__)

# Load model
model = pickle.load(open('Final_Model.sav', 'rb'))

# Return top words from txt file
def get_top_words():
    with open("top_words.txt") as file:
        lines = [line.rstrip() for line in file]
    return lines

# Home page route
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

# Predict page route
@app.route('/', methods=['POST'])
def predict():
    msg = request.form.get('msg')
    msg = cleaning_process(msg)
    corpus = [nltk.word_tokenize(msg.lower())]
    top_words = get_top_words()
    bow_features = bow_for_line(corpus, top_words)
    print(corpus)
    pred = model.predict(bow_features)
    print(pred)
    if pred[0] == 1:
        pred = 'Depression'
    else:
        pred = 'Not depression'

    return render_template('index.html', prediction=pred)

# Run localhost server
if __name__ == "__main__":
    app.run(port=3000, debug=True)
