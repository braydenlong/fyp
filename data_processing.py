import numpy as np
import pandas as pd
import string
import emoji
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from afinn import Afinn
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD



#import data
df = pd.read_csv("ori.csv")

#reading datalabels
print(df.columns)


#helper functions

#Nonverbal Feature - Verified Review
def verified(status):
    if status == 'N':
        return 0
    else:
        return 1

#Preprocessing Techniques for Sentiment Analysis: FIND DETAILED OF ALL
def clean_text(text):
    # text = re.sub(r'[^\w\s]','',text) - remove special characters
    text = text.lower() # lowercase the text
    text = word_tokenize(text) #tokenise text
    text = [word for word in text if word not in stopwords.words('english')] # remove stopwords
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text] # perform lemmatization
    text = " ".join(text)
    return text


#Verbal Feature - Word Count
def get_word_counts(text):
    words = word_tokenize(text)
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return sum(word_counts.values())

#Verbal Feature - Caps Count
def count_caps(text):
    caps = sum(1 for char in text if char.isupper())
    return caps

#Verbal Feature - Punct Count
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return count


#Verbal Feature - Emojis behavioural features
def count_emojis(text):
    # Define a regular expression to match emoji characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # other symbols
        u"\U000024C2-\U0001F251" 
                           "]+", flags=re.UNICODE)
    # Count the number of matches in the text
    emoji_count = len(emoji_pattern.findall(text))
    return emoji_count

def sentiment_score(text):
    return Afinn().score(text)






#data preprocessing
#obtain required columns from the dataframe
df_updated = df[['LABEL','RATING','VERIFIED_PURCHASE','PRODUCT_CATEGORY','PRODUCT_ID','PRODUCT_TITLE','REVIEW_TITLE','REVIEW_TEXT']].copy()
df_updated['TOTAL_TEXT'] = df_updated.apply(lambda x: x['REVIEW_TITLE'] + "-" + x['REVIEW_TEXT'], axis=1)
df_updated.drop(['REVIEW_TITLE','REVIEW_TEXT'], axis=1, inplace=True)

df_updated = df_updated[['LABEL','RATING','VERIFIED_PURCHASE','PRODUCT_CATEGORY','PRODUCT_ID','PRODUCT_TITLE','TOTAL_TEXT']]

#TO CONSIDER: drop 'DOC_ID' column, maybe drop product, product id <NO!!!!!><BECAUSE OF AMAZONREVIEWCHECKER>
df_updated = df_updated.drop(['PRODUCT_CATEGORY','PRODUCT_ID','PRODUCT_TITLE'], axis=1)
# print('line 115')
# print(df_updated)

#******
# based on raw data, there are only (label, rating, verified purchase, total text)
#******

#one-hot encoding on label for real or fake review
one_hot_real_or_fake = pd.get_dummies(df_updated['LABEL'])
df_updated = pd.concat([df_updated, one_hot_real_or_fake], axis=1)
df_updated.drop('LABEL', axis=1, inplace=True)
df_updated = df_updated.rename(columns={'__label1__': 'is_fake'})
df_updated.drop('__label2__', axis=1, inplace = True)

# print('line 127')
# print(df_updated)


#Convert verified purchase to numerical
df_updated['VERIFIED_PURCHASE'] = df_updated['VERIFIED_PURCHASE'].apply(lambda x:verified(x))


#apply NLP on total_text (review title and review text)

df_updated['text_cleaned'] = df_updated['TOTAL_TEXT'].apply(lambda x: clean_text(x))
df_updated.dropna(inplace = True) #remove rows with missing values
df_updated = df_updated[df_updated['text_cleaned'] != ""] #remove rows with empty text

word_counts = df_updated['text_cleaned'].apply(lambda x: get_word_counts(x)) # Get word counts
df_updated['word_count'] = word_counts
cap_counts = df_updated['text_cleaned'].apply(lambda x: count_caps(x)) # Get caps counts
df_updated['caps_count'] = cap_counts
punct_counts = df_updated['text_cleaned'].apply(lambda x: count_punct(x)) # Get punct counts
df_updated['punct_count'] = punct_counts
emoji_counts = df_updated['text_cleaned'].apply(lambda x: count_emojis(x)) # Get emoji counts
df_updated['emoji_count'] = emoji_counts
sentiment_value = df_updated['text_cleaned'].apply(lambda x: sentiment_score(x)) # get sentiment score
df_updated['sentiment_score'] = sentiment_value
# print('line 151')
# print(df_updated)

vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df_updated['text_cleaned']) # Convert text to a bag of words representation
# df_vectorized = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())
# df_updated = pd.concat([df_updated, df_vectorized], axis=1)
# print('line 158')
# print(df_updated)

model = LatentDirichletAllocation(n_components=10)

model_result = model.fit_transform(vectorized) # Fit a Latent Dirichlet Allocation (LDA) model: to extract topics
df_model = pd.DataFrame(model_result, columns=['topic_' + str(i) for i in range(10)])
# print(df_updated)
# print(df_model)
df_updated = pd.concat([df_updated, df_model], axis=1)
# print('line 168')
# print(df_updated)



# def get_top_n_words(model, feature_names, n=10):
#     topic_word_distributions = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
#     topic_words = [feature_names[i] for i in np.argsort(topic_word_distributions, axis=1)[:,:-n-1:-1]]
#     return topic_words

#top_words = get_top_n_words(model, feature_names) # Get the top 10 words for each topic

#tf-idf
vectorizer = TfidfVectorizer()
tfidf_text = vectorizer.fit_transform(df_updated['text_cleaned'])
#feature_names = vectorizer.get_feature_names_out()

# denselist = []
# batch_size = 1000
# num_batches = (tfidf_text.shape[0] + batch_size - 1) // batch_size
# for i in range(num_batches):
#     start = i * batch_size
#     end = min((i+1) * batch_size, tfidf_text.shape[0])
#     batch = tfidf_text[start:end].toarray().tolist()
#     denselist.extend(batch)

svd = TruncatedSVD(n_components=50)
svd_data = svd.fit_transform(tfidf_text)
svd_df = pd.DataFrame(svd_data)
df_updated = pd.concat([df_updated, svd_df], axis=1)

# tf_idf_df = pd.DataFrame(denselist, columns=feature_names)
# df_updated = pd.concat([df_updated, tf_idf_df], axis=1)
# df_updated = df_updated.astype({col: 'float' for col in tf_idf_df.columns})

print("line 186")
print(df_updated.shape[1])

#save the processed dataset
df_updated.to_csv('processed_data.csv', index = False)

#alternative feature extraction technique from tf-idf: word embeddings

# bi-tri gram models