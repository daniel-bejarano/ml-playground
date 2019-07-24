'''
    PROJECT STATEMENT:
        To create a ML pipeline that is able to classify headlines as
        coming from Left, Right or Center sources in the political spectrum.

    FILE STATEMENT:
        This file performs the following processes:
            1. Load headlines datasets that were previously captured and saved into computer
            2. Train ML model to classify headlines and save model to be used by other files
'''


### Import Libraries ###
import pandas as pd
import os
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pickle

##### LOAD HEADLINES #####
this_dir = os.path.dirname(os.path.abspath("__file__"))
dump_dir = os.path.join(this_dir, "Extracted Data")
headlines_df = pd.DataFrame()

for fname in os.listdir(dump_dir):  # function to iterate over, and combine all saved csv files with headlines
    file_path = os.path.join(dump_dir, fname)
    df = pd.read_csv(file_path, encoding="utf8")
    headlines_df = pd.concat([headlines_df, df], axis=0, sort=True)

# Drop Nan values and remove duplicates
headlines_df.drop("Unnamed: 0", inplace=True, axis=1)
headlines_df.drop_duplicates(inplace=True)

# Shuffle headlines to avoid structure into the order of our dataset
shuffled_headlines_df = headlines_df.sample(len(headlines_df))


##### PROCESS TEXT #####

# Define function to process text
def text_process(mess):# mess stands for message
    '''
    1. Remove punctuation
    2. Remove Stop Words
    3. Return list of cleaned text words
    '''
    nopunc = [c for c in mess if c not in string.punctuation]
    nopunc = "".join(nopunc) # to join together separated characters back into words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in nopunc.split() if w.lower() not in stopwords.words('english')]
        # the above returns a list of list of words
    cleaned_sent = " ".join(lemmatized_words) # returns a list of sentences
    return(cleaned_sent)

# Process Text
clean_headlines = [text_process(mess) for mess in shuffled_headlines_df['Headlines']]


##### DATA PROCESSING #####

# Convert Inclinations to Categories and Define y as numerical labels
shuffled_headlines_df['Inclination'] = pd.Categorical(shuffled_headlines_df['Inclination'])
y = shuffled_headlines_df['Inclination'].cat.codes
labels_dict = dict(enumerate(shuffled_headlines_df['Inclination'].cat.categories))

# Split into Train and Test
headlines_train, headlines_test, y_train, y_test = train_test_split(clean_headlines, y, test_size=0.25)


##### TF-IDF SCORES #####


def get_word_scores(train_data, test_data, vectorizer):  # Define Function to Get TF-IDF Scores and Counts
    ''' Function to vectorize data and get words scores. In this particular case
    we are using it for tf-idf scores and word counts.
    The reason to include the test data, which is not immediately apparent, is that one of the classifiers in our
    sentiment analysis will be based off the tf-idf and count vectors.
    Input: training list of sentences, test list of sentences, and the vectorizer to use (e.g. CountVectorizer)
    Outpout: train and test observations in tf-idf scores form, word score, and the fitted vectorizer '''

    # Fit vectorizer to train data. Transform train and test sets
    vectorizer.fit(train_data)
    x_train = vectorizer.transform(train_data)
    x_test = vectorizer.transform(test_data)

    # Highest TF-IDF Scores
    vocab = vectorizer.vocabulary_  # gives the number of words in our vocabulary
    sum_words = x_train.sum(axis=0)
    words_freq = sorted([(word, sum_words[0, idx]) for word, idx in vocab.items()],
                        key=lambda x: x[1], reverse=True)
    return (x_train, x_test, words_freq, vectorizer)


# Get TF-IDF Score Vectors on Headliens
tf_vectorizer = TfidfVectorizer(min_df=3,
                                max_features=1000,
                                strip_accents='unicode',
                                analyzer='word',
                                ngram_range=(1, 1),
                                use_idf=1,
                                smooth_idf=1,
                                sublinear_tf=1,
                                stop_words='english')

X_train_tf, X_test_tf, tf_scores, tf_model = get_word_scores(headlines_train, headlines_test, tf_vectorizer)


##### TRAIN LOGISTIC REGRESSION MODEL #####

# Train Model on Training Set
model = LogisticRegression()
model.fit(X_train_tf, y_train)

# Predict on Test Set
y_pred = model.predict(X_test_tf)

# Create Test Report
print('Classification Report: ', '\n', classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(labels_dict)


##### TRAIN LOGISTIC REGRESSION ON ENTIRE DATASET #####

# Combine train and test sets of X and y
all_X_tf, _, tf_scores, tf_model = get_word_scores(headlines_train + headlines_test, headlines_test, tf_vectorizer)
all_y = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)

# Train model on combined X and y
log_reg = LogisticRegression()
final_classifier = log_reg.fit(all_X_tf, all_y)

with open('final_classifier.pkl', 'wb') as fid:
    pickle.dump(final_classifier, fid)

with open('tf-idf_vectorizer.pkl', 'wb') as fid:
    pickle.dump(tf_model, fid)

with open('labels_dictionary.pkl', 'wb') as fid:  # Save dictionary containing relationship between label and y
    pickle.dump(labels_dict, fid)

