'''
    PROJECT STATEMENT:
        To create a ML pipeline that is able to classify headlines as
        coming from Left, Right or Center sources in the political spectrum.

    FILE STATEMENT:
        This file performs the following processes:
            1. Create online interaction via Flask to allow users to enter a headline and obtain a label
'''

##### LOAD LIBRARIES #####
import pickle
from flask import Flask, request, render_template, redirect, url_for, flash, redirect
from forms import HeadlineForm
from sklearn.externals import joblib

##### SERVER INTERFACE #####

app = Flask(__name__)
app.config['SECRET_KEY'] = 'd7167d24cf9014730b997edae5f05873'  # Code to prevent cookies from pulling information


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', title='Home Page')


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/headline_classifier", methods=['GET', 'POST'])
def headline_classifier():
    form = HeadlineForm()
    if form.validate_on_submit():

        ##### LOAD VECTORIZER & CLASSIFIER #####
        tf_vectorizer = joblib.load('tf-idf_vectorizer.pkl')
        final_classifier = joblib.load('final_classifier.pkl')
        labels_dict = joblib.load('labels_dictionary.pkl')

        ##### PREDICT LABEL ON NEW HEADLINE #####
        # Transform Headline to TF-IDF Score
        news_headline = form.headline.data
        tf_headline = tf_vectorizer.transform([news_headline])

        # Predict on user-entered headlines
        news_headline_pred = final_classifier.predict(tf_headline)
        news_headline_label = str(labels_dict[int(news_headline_pred)])

        # Print Result
        flash(f'The headline "{news_headline}" is classified as: \n {news_headline_label.upper()}')

    return render_template('headline.html', title='Classifier', form=form)


if __name__ == '__main__':  # Checks that we only run the app when this file is being called directly
    app.run(debug=False)
# TODO: Set debug to False