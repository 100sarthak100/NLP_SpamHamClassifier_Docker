import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from flask import Flask, render_template, url_for, request

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)

# # make a model
# df = pd.read_csv("spam.csv", encoding="latin-1")
# df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# # features and labels
# df['label'] = df['class'].map({'ham': 0, 'spam': 1})
# X = df['message']
# y = df['label']

# # extract feature with CountVectorizer
# cv = CountVectorizer()
# X = cv.fit_transform(X)

# pickle.dump(cv, open('transform.pkl', 'wb'))

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Naive bayes classifier
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# clf.fit(X_train, y_train)
# clf.score(X_test, y_test)
# pickle.dump(clf, open('nlp_model.pkl', 'wb'))








