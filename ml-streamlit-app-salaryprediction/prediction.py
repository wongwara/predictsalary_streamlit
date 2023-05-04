import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/wongwara/Jobseeker_Baymax/main/dataset/final_cleaned.csv', index_col=[0])

#code for encoder only the columns that need to transform
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
jobClassification_cats = [['Information & Communication Technology',
       'Banking & Financial Services', 'Science & Technology',
       'Education & Training', 'Government & Defence',
       'Consulting & Strategy', 'Healthcare & Medical',
       'Human Resources & Recruitment', 'Marketing & Communications',
       'Retail & Consumer Products', 'Administration & Office Support',
       'Accounting', 'Insurance & Superannuation',
       'Mining, Resources & Energy', 'Real Estate & Property',
       'Manufacturing, Transport & Logistics', 'Engineering']]

state_cats = [['Australian Capital Territory', 'South Australia','Western Australia']]
df = df.drop(['state_encoded','min_salary','max_salary','workType','salary_section_enc'],axis =1)

df['desktopAdTemplate']= df['desktopAdTemplate'].fillna('')
df['teaser']= df['teaser'].fillna('')

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Clean the text data
df['teaser'] = df['teaser'].str.replace('[^\w\s]', '') # Remove punctuation
df['desktopAdTemplate'] = df['desktopAdTemplate'].str.replace('[^\w\s]', '') # Remove punctuation
df['teaser'] = df['teaser'].str.replace('\d+', '') # Remove digits
df['desktopAdTemplate'] = df['desktopAdTemplate'].str.replace('\d+', '') # Remove digits

# Normalize the text data
stop_words = set(stopwords.words('english'))
df['teaser'] = df['teaser'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_words]))
df['desktopAdTemplate'] = df['desktopAdTemplate'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_words]))

# Tokenize the text data
df['teaser'] = df['teaser'].apply(lambda x: word_tokenize(x))
df['desktopAdTemplate'] = df['desktopAdTemplate'].apply(lambda x: word_tokenize(x))

# Apply stemming
stemmer = PorterStemmer()
df['teaser'] = df['teaser'].apply(lambda x: [stemmer.stem(word) for word in x])
df['desktopAdTemplate'] = df['desktopAdTemplate'].apply(lambda x: [stemmer.stem(word) for word in x])

# # Create TF-IDF vectors
vectorizer = TfidfVectorizer()
teaser_tfidf = vectorizer.fit_transform(df['teaser'].apply(lambda x: ' '.join(x)))
desktopAdTemplate_tfidf = vectorizer.fit_transform(df['desktopAdTemplate'].apply(lambda x: ' '.join(x)))

# # Concatenate the TF-IDF vectors with the original dataframe
df = pd.concat([df.drop(['teaser', 'desktopAdTemplate'], axis=1), pd.DataFrame(teaser_tfidf.toarray()), pd.DataFrame(desktopAdTemplate_tfidf.toarray())], axis=1)

# Display the resulting dataframe
print(df.head())

# Separate to X and y
y = df.pop('salary_section')
X = df
from sklearn.model_selection import train_test_split
# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.columns = X_train.columns.astype(str) 
X_test.columns = X_test.columns.astype(str)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
# Model building

# create an SVM model with the best hyperparameters found using grid search
svm_model = SVC(C=10, gamma='scale', kernel='linear')

# fit the SVM model to the training data
svm_model.fit(X_train, y_train)

# make predictions on the test data using the trained SVM model
y_pred = svm_model.predict(X_test)

# calculate the accuracy of the SVM model predictions on the test data
accuracy = accuracy_score(y_test, y_pred)

def preprocess_text_input(input_str):
    # Clean the text data
    input_str = input_str.replace('[^\w\s]', '') # Remove punctuation
    input_str = input_str.replace('\d+', '') # Remove digits
    # Normalize the text data
    stop_words = set(stopwords.words('english'))
    input_str = ' '.join([word.lower() for word in input_str.split() if word.lower() not in stop_words])
    # Tokenize the text data
    input_str = word_tokenize(input_str)
    # Apply stemming
    stemmer = PorterStemmer()
    input_str = [stemmer.stem(word) for word in input_str]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    input_tfidf = vectorizer.fit_transform([' '.join(input_str)])

    return input_tfidf.toarray()


jobClassification_enc = OrdinalEncoder(categories=jobClassification_cats)


import pickle
data = {"model": svm_model, "encode": jobClassification_enc}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

regressor_loaded = data["model"]
jobClassification_enc = data["encode"].values.reshape(-1, 1)

# y_pred = regressor_loaded.predict(X)
# y_pred
