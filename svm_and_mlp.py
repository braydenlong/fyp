import numpy as np
import pandas as pd
import csv
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
#imports above not used?
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import torch.nn as nn
#from torchsvm import WeightedSVM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import data
df_updated = pd.read_csv("processed_data.csv")
print(df_updated.shape[1])

#-----------MODELS AND 10-FOLD CROSS VALIDATION---------------
#-----------HYPERPARAMETER TUNING VIA GRID SEARCH ------------
#-----------SAVING OF EACH RESULT IN JOBLIB FILE--------------

#splitting of data
X = df_updated
scaler = MinMaxScaler(feature_range=(0, 1))
X['sentiment'] = scaler.fit_transform(X[['sentiment_score']])
X_scaled = scaler.fit_transform(X.iloc[:, 21:])
X.iloc[:, 21:] = X_scaled

X = X.drop(['is_fake','TOTAL_TEXT','text_cleaned','sentiment_score'], axis=1)
y = df_updated['is_fake']
print(X.dtypes)

X_np = np.array(X)
y_np = np.array(y)

X_train, X_valid, y_train, y_valid = train_test_split(X_np,y_np,train_size=0.8, test_size=0.2, random_state=None)
kf = KFold(n_splits=10, shuffle=True, random_state=42)


X_tensor = torch.tensor(X_np).float().to(device)
y_tensor = torch.tensor(y_np).float().to(device)
X_train_tensor = torch.tensor(X_train).float().to(device)
y_train_tensor = torch.tensor(y_train).float().to(device)
X_valid_tensor = torch.tensor(X_valid).float().to(device)
y_valid_tensor = torch.tensor(y_valid).float().to(device)


#SVM <----TO DELETE??--->

# svm = SVC()


# param_grid = {'C': [0.1, 1, 10, 100],'gamma': [0.1, 1, 10, 100],
#               'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

# grid_search = GridSearchCV(svm, param_grid, cv=kf)
# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_estimator_
# y_pred = grid_search.best_estimator_.predict(X_valid)


# print("SVM")
# print(classification_report(y_valid, y_pred))
# print("Accuracy:", accuracy_score(y_valid, y_pred))

# scores = cross_val_score(grid_search.best_estimator_, X, y, cv=kf)
# mean_score = np.mean(scores)
# std_score = np.std(scores)

# print(f"Mean accuracy score: {mean_score:.2f}")
# print(f"Standard deviation: {std_score:.2f}")
# print("************************")

# with open('results.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow({
#         'Model': 'SVM',
#         'Parameters': str(best_params),
#         'Accuracy': accuracy_score(y_valid, y_pred),
#         'Mean Score': mean_score,
#         'Std Score': std_score
#     })

# joblib.dump(best_params, 'best_params/svm_best_params.pkl')


#MLP

param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['logistic', 'relu'],
    'alpha': [0.0001, 0.001, 0.01],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive'],
    'batch_size': [32, 64, 128],
    'max_iter': [500, 1000, 2000]
}

mlp = MLPClassifier()
grid_search = GridSearchCV(mlp, param_grid, cv=kf)
grid_search.fit(X_train_tensor, y_train_tensor)
best_mlp = grid_search.best_estimator_

#best_mlp.to(device)

y_pred_tensor = best_mlp.predict(X_valid_tensor)
#y_pred = y_pred_tensor.numpy()

# mlp.fit(X_train, y_train)
# y_pred = mlp.predict(X_valid)
print("Multi Layer Perceptron")
print(classification_report(y_valid_tensor, y_pred_tensor))
print("Accuracy: ", accuracy_score(y_valid_tensor, y_pred_tensor))

scores = cross_val_score(mlp, X_tensor, y_tensor, cv=kf)
mean_score = np.mean(scores)
std_score = np.std(scores)

print(f"Mean accuracy score: {mean_score:.2f}")
print(f"Standard deviation: {std_score:.2f}")
print("************************")

with open('results.csv', 'a', newline='') as csvfile:
    fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({
        'Model': 'MLP',
        'Parameters': str(best_mlp),
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Mean Score': mean_score,
        'Std Score': std_score
    })
joblib.dump(best_mlp, 'best_params/mlp_best_params.pkl')