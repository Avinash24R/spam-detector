from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract Feature With CountVectorizer
    cv = pickle.load(open("vectorizer.pk1", 'rb'))
    # Naive Bayes Classifier
    clf = pickle.load(open("model.pkl", 'rb')) 
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run()