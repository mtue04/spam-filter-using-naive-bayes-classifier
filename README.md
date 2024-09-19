# Spam Detection using Naive Bayes Classifier

This project implements a machine learning model to detect spam messages using Natural Language Processing (NLP) techniques and a Naive Bayes classifier. The main goal of this project is to classify whether a given message is spam or not based on textual features extracted from the message content.

## Project Overview
This project preprocesses text data from a dataset of messages, applies feature extraction, and uses a Multinomial Naive Bayes classifier to train a model that can classify spam messages. The main steps involved are:

- Text preprocessing (removing special characters, stopwords, stemming)
- Vectorization of text using TfidfVectorizer
- Feature selection using SelectKBest
- Model training and evaluation using a Naive Bayes classifier
- Cross-validation and hyperparameter tuning using GridSearchCV
- Making predictions on new messages to classify them as spam or not spam

## About Dataset

### Context
The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

### Content
The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.

This corpus has been collected from free or free for research sources at the Internet:

-> A collection of 425 SMS spam messages was manually extracted from the Grumbletext Web site. This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of the text of spam messages in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. \
-> A subset of 3,375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC), which is a dataset of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available. \
-> A list of 450 SMS ham messages collected from Caroline Tag's PhD Thesis. \
-> Finally, we have incorporated the SMS Spam Corpus v.0.1 Big. It has 1,002 SMS ham messages and 322 spam messages. This corpus has been used in the following academic researches:

### Acknowledgements
The original dataset can be found here. The creators would like to note that in case you find the dataset useful, please make a reference to previous paper and the web page: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/ in your papers, research, etc.

We offer a comprehensive study of this corpus in the following paper. This work presents a number of statistics, studies and baseline results for several machine learning methods.

Almeida, T.A., GÃ³mez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011.

## Installation
To run this project, follow these steps:

Clone the repository:
```
git clone https://github.com/mtue04/spam-filter-using-naive-bayes-classifier.git
cd spam-filter-using-naive-bayes-classifier
```

Download the required NLTK data (this is handled automatically in the code):
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage
### Training the Model
1. Preprocess your text data and prepare it for training by applying the text preprocessing functions.
2. Train the Naive Bayes classifier by splitting your dataset into training and test sets. You can customize the vectorization and feature selection parameters as needed.
3. Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score. \
`best_model, misclassified_samples = train_and_evaluate_naive_bayes(X_train_selected, y_train)`

### Making Predictions
To make predictions on new messages (e.g., from user input or a different dataset), use the `predict_spam` function:
```
new_message = 'Your cash-balance is currently 500 pounds - to maximize your cash-in now, send COLLECT to 83600.'

result, prediction_rate = predict_spam(new_message)
print(f"Prediction: {result} with {prediction_rate}%")
```
