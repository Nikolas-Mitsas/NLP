#MACHINE LEARNING CHALLENGE
#880083-M-6 MACHINE LEARNING
#GRZEGORZ CHRUPAŁA
#12 DECEMBER 2022

#Anthonijsz, Romy | Snr. 2030456
#Drenth, Joya | Snr. 2001662
#Mitsas, Nikolas | Snr. 2090834 
#Rivera, Alejandro | 2097973

#START CODE

#IMPORTING ALL NECESSARY LIBRARIES AND FUNCTIONS
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder

#CREATE A FUNCTION THAT PREPROCESSES A TEXT FEATURE. REMOVING PUNCTUATION, MAKING WORDS LOWERCASE
def lem(abstract):
    ab_list = abstract.strip().lower().split()
    new_ab = []
    for word in ab_list:
        new_word = ""
        for letter in word:
            if letter.isalpha() or letter.isdigit():
                new_word += letter
        new_ab.append(new_word)
    filtereed_word = [w for w in new_ab if not w in stop_words]
    new_abstract = " ".join(filtereed_word)
    return new_abstract


#READ IN THE TRAIN JSON FILE AND ASSIGNING IT TO A VARIABLE. TURN IT INTO A PANDAS DATAFRAME FOR EASE
f_train = open("train.json")
file_train = json.load(f_train)
data_train = pd.DataFrame(file_train)
y = data_train['authorId']

#INSPECTING THE DATA/EDA
print(data_train.head())
print(y.head())

#READ IN THE TEST JSON FILE AND ASSIGN IT TO A VARIABLE AS WELL AND TURN IT INTO A PANDAS DATAFRAME FOR EASE
f_test = open("test.json")
file_test = json.load(f_test)
data_test = pd.DataFrame(file_test)

#INSPECT THE FIRST COUPLE ROWS OF THIS DATAFRAME TOO
data_test.head()

#ASSIGN FUNCTIONS TO VARIABLES FOR READABILITY WHEN THE FUNCTIONS ARE USED
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#COMBINING THE TEXT FEATURES
data_train['all'] = data_train['title'] + " " + data_train['abstract']

#FIRST OUR LEM FUNCTION IS APPLIED TO THE ABSTRACT FEATURE IN THE TRAIN DATASET MAKING THE WORDS EASIER TO PROCESS
data_train['lemmatizer'] = data_train['all'].apply(lem)

#THE LEMMATIZER FUNCTION IS THEN APPLIED BRINGING THE WORDS TO THEIR BASE TO PREVENT UNNECESSARY OVERLAP BETWEEN WORDS WITH THE SAME BASE
data_train['result'] = data_train['lemmatizer'].apply(lambda x: "".join([lemmatizer.lemmatize(i) for i in x]))

#FUNCTION THAT TURNS THE TEXT SUPPLIED INTO FEATURES AND COUNTS HOW OFTEN EACH TERM OCCURS
#MAX FEATURES MEANS ONLY 31000 TERMS, IN THIS CASE, ARE USED.
#N GRAM RANGE MEANS THAT THE TERMS CAN BE COMBINED INTO SETS OF MAX 3
count_vectorizer = CountVectorizer(max_features=31000, ngram_range=(1, 3))

#COUNT VECTORIZER IS APPLIED TO THE PREPROCESSED TEXT FEATURE
count_train = count_vectorizer.fit_transform(data_train['result'])
count_train = pd.DataFrame(count_train.toarray())

#INSPECTING WHAT THE DATA LOOKS LIKE
count_train.shape
count_train.head()


#ONE HOT ENCODING THE VENUE AND THE YEAR FEATURES
oh = OneHotEncoder(handle_unknown='ignore')
X_train_num = oh.fit_transform(data_train[['venue', 'year']])

#COMBINING THE ENCODED FEATURES WITH THE REST OF THE DATASET
train_hot = pd.DataFrame(X_train_num.toarray())
count_train = pd.concat([count_train, train_hot], axis=1)

#CHECKING THE SHAPE TO SEE HOW MANY NEW FEATURES WERE CREATED
count_train.shape

#ABSTRACT AND TITLE FEATURE COMBINED FOR THE TEST DATA
data_test['all'] = data_test['title'] + " " + data_test['abstract']

#LEM, LEMMATIZER, AND COUNTVECTORIZER FUNCTIONS APPLIED TO THE TEST DATA THIS TIME
data_test['lemmatizer'] = data_test['all'].apply(lem)
data_test['result'] = data_test['lemmatizer'].apply(lambda x: "".join([lemmatizer.lemmatize(i) for i in x]))
count_test = count_vectorizer.transform(data_test['result'])
count_test = pd.DataFrame(count_test.toarray())

#ONE HOT ENCODING THE VENUE AND THE YEAR FEATURES OF THE TEST DATA
X_test_num = oh.transform(data_test[['venue', 'year']])
test_hot = pd.DataFrame(X_test_num.toarray())

#COMBINING THE ENCODED FEATURES WITH THE REST OF THE DATASET
count_test = pd.concat([count_test, test_hot], axis=1)
#CHECKING THE NEW FEATURES
count_test.shape
count_test

#MULTINOMIAL NAIVE BAYES MODEL IS TRAINED USING THE PREPARED TRAINING DATA
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB(alpha=0.01)
nb_classifier.fit(count_train, y)
pred_freq = nb_classifier.predict(count_test)

#COMBINING THE PREDICTIONS AND THE AUTHOR NAMES
last = pd.DataFrame(pred_freq, columns=["authorId"])
print(last.head())

results = pd.concat([data_test["paperId"], last], axis=1)
print(results.head())

#CREATING A DICTIONARY WITH THE PREDICTIONS
prediction_list = []
for i in range(len(results)):
    predictions_dict = {}
    predictions_dict["paperId"] = str(results["paperId"][i])
    predictions_dict["authorId"] = str(results["authorId"][i])
    prediction_list.append(predictions_dict)
print(prediction_list)


#WRITING THE PREDICTIONS TO A JSON FILE TO UPLOAD IT
with open('predicted.json', 'w') as out_file:
     json.dump(prediction_list, out_file)

out_file.close()

