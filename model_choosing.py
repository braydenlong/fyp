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
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8, test_size=0.2, random_state=None)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

#random forest <----FIFTH---->
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# rfc = RandomForestClassifier(random_state = 40)
# grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=kf, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_


# rfc = RandomForestClassifier(**best_params, random_state=42)
# rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_valid)
# print("Random Forest")
# print(classification_report(y_valid, y_pred))
# print("Accuracy", accuracy_score(y_valid, y_pred))

# scores = cross_val_score(rfc, X, y, cv=kf)
# mean_score = np.mean(scores)
# std_score = np.std(scores)
# print(f"Mean accuracy score: {mean_score:.2f}")
# print(f"Standard deviation: {std_score:.2f}")
# print("************************")

# with open('results.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow({
#         'Model': 'Random Forest',
#         'Parameters': str(best_params),
#         'Accuracy': accuracy_score(y_valid, y_pred),
#         'Mean Score': mean_score,
#         'Std Score': std_score
#     })

# joblib.dump(best_params, 'rf_best_params.pkl')

# #SVM <----TO DELETE??--->

svm = SVC()
param_grid = {'C': [0.1],
              'kernel': ['linear']}

# param_grid = {'C': [0.1, 1, 10, 100],'gamma': [0.1, 1, 10, 100],
#               'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

grid_search = GridSearchCV(svm, param_grid, cv=kf)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_estimator_
y_pred = grid_search.best_estimator_.predict(X_valid)


print("SVM")
print(classification_report(y_valid, y_pred))
print("Accuracy:", accuracy_score(y_valid, y_pred))

scores = cross_val_score(grid_search.best_estimator_, X, y, cv=kf)
mean_score = np.mean(scores)
std_score = np.std(scores)

print(f"Mean accuracy score: {mean_score:.2f}")
print(f"Standard deviation: {std_score:.2f}")
print("************************")

with open('results.csv', 'a', newline='') as csvfile:
    fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({
        'Model': 'SVM',
        'Parameters': str(best_params),
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Mean Score': mean_score,
        'Std Score': std_score
    })

joblib.dump(best_params, 'best_params/svm_best_params.pkl')


#logistic regression <-----SIXTH----->
# param_grid = {
#     'penalty': ['l1', 'l2'],
#     'C': [0.1, 1, 10],
# }

# regressor = LogisticRegression(max_iter=10000, solver='saga')

# grid_search = GridSearchCV(regressor, param_grid, cv=kf)
# grid_search.fit(X_train, y_train)

# best_logreg = grid_search.best_estimator_
# # regressor.fit(X_train, y_train)
# # y_pred = regressor.predict(X_valid)
# y_pred = best_logreg.predict(X_valid)

# print("Logistic Regression")
# print(classification_report(y_valid, y_pred))
# print(accuracy_score(y_valid, y_pred))

# scores = cross_val_score(regressor, X, y, cv=kf)
# mean_score = np.mean(scores)
# std_score = np.std(scores)

# print(f"Mean accuracy score: {mean_score:.2f}")
# print(f"Standard deviation: {std_score:.2f}")
# print("************************")

# with open('results.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow({
#         'Model': 'Logistic Regression',
#         'Parameters': str(best_logreg),
#         'Accuracy': accuracy_score(y_valid, y_pred),
#         'Mean Score': mean_score,
#         'Std Score': std_score
#     })

# joblib.dump(best_logreg, 'lr_best_params.pkl')

# # #gradient-boosting <----FOURTH--->
# param_grid = {'n_estimators': [50, 100, 200],
#               'learning_rate': [0.01, 0.1, 0.5],
#               'max_depth': [1, 2, 3, 4]}

# grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42),
#                            param_grid=param_grid,
#                            cv=kf,
#                            n_jobs=-1)

# grid_search.fit(X_train, y_train)
# best_gb = grid_search.best_estimator_
# y_pred = best_gb.predict(X_valid)

# gradientbooster = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
# gradientbooster.fit(X_train, y_train)
# y_pred = gradientbooster.predict(X_valid)

# print("Gradient Boosting")
# print(classification_report(y_valid, y_pred))
# print("Accuracy", accuracy_score(y_valid, y_pred))


# scores = cross_val_score(best_gb, X, y, cv=kf)
# mean_score = np.mean(scores)
# std_score = np.std(scores)

# print(f"Mean accuracy score: {mean_score:.2f}")
# print(f"Standard deviation: {std_score:.2f}")
# print("************************")

# with open('results.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow({
#         'Model': 'Gradient Booster',
#         'Parameters': str(best_gb),
#         'Accuracy': accuracy_score(y_valid, y_pred),
#         'Mean Score': mean_score,
#         'Std Score': std_score
#     })

# joblib.dump(best_gb, 'best_params/gb_best_params.pkl')

# #decision-tree <-----THIRD------>
# param_grid = {'max_depth': [2, 4, 6, 8],
#               'min_samples_split': [2, 4, 6, 8]}
# dectreeclf = DecisionTreeClassifier()
# grid_search = GridSearchCV(dectreeclf, param_grid=param_grid, cv=kf)
# grid_search.fit(X_train, y_train)

# best_clf = grid_search.best_estimator_
# y_pred = best_clf.predict(X_valid)

# # dectreeclf.fit(X_train, y_train)
# # y_pred = dectreeclf.predict(X_valid)
# print("Decision Tree")
# print(classification_report(y_valid, y_pred))
# print("Accuracy",accuracy_score(y_valid, y_pred))

# scores = cross_val_score(best_clf, X, y, cv=kf)
# mean_score = np.mean(scores)
# std_score = np.std(scores)

# print(f"Mean accuracy score: {mean_score:.2f}")
# print(f"Standard deviation: {std_score:.2f}")
# print("************************")

# with open('results.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow({
#         'Model': 'Decision Tree',
#         'Parameters': str(best_clf),
#         'Accuracy': accuracy_score(y_valid, y_pred),
#         'Mean Score': mean_score,
#         'Std Score': std_score
#     })

# joblib.dump(best_clf, 'dectree_best_params.pkl')


# #Naive Bayes <----FIRST--->
# param_grid = {'alpha': [0.1, 1.0, 10.0]}
# nb_clf = MultinomialNB()
# grid_search = GridSearchCV(nb_clf, param_grid=param_grid, cv=kf)
# grid_search.fit(X_train, y_train)

# best_nb = grid_search.best_estimator_
# y_pred = best_nb.predict(X_valid)


# # nb_clf.fit(X_train, y_train)
# # y_pred = nb_clf.predict(X_valid)

# print("Naive Bayes")
# print(classification_report(y_valid, y_pred))
# print(accuracy_score(y_valid, y_pred))

# scores = cross_val_score(best_nb, X, y, cv=kf)
# mean_score = np.mean(scores)
# std_score = np.std(scores)

# print(f"Mean accuracy score: {mean_score:.2f}")
# print(f"Standard deviation: {std_score:.2f}")
# print("************************")

# joblib.dump(best_nb, 'nb_best_params.pkl')

# with open('results.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow({
#         'Model': 'NB',
#         'Parameters': str(best_nb),
#         'Accuracy': accuracy_score(y_valid, y_pred),
#         'Mean Score': mean_score,
#         'Std Score': std_score
#     })

#KNN <----------------SECOND------->
# param_grid = {'n_neighbors': [3, 5, 7, 9],
#               'metric': ['euclidean', 'manhattan', 'minkowski']}

# knn = KNeighborsClassifier()
# grid_search = GridSearchCV(knn, param_grid, cv=kf, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# best_knn = grid_search.best_estimator_
# y_pred = best_knn.predict(X_valid)

# # knn.fit(X_train, y_train)
# # y_pred = knn.predict(X_valid)
# print("K Nearest Neighbour")
# print(classification_report(y_valid, y_pred))
# print("Accuracy:", accuracy_score(y_valid, y_pred))

# scores = cross_val_score(best_knn, X, y, cv=kf)
# mean_score = np.mean(scores)
# std_score = np.std(scores)

# print(f"Mean accuracy score: {mean_score:.2f}")
# print(f"Standard deviation: {std_score:.2f}")
# print("************************")

# joblib.dump(best_knn, 'knn_best_params.pkl')
# print(best_knn)

# with open('results.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow({
#         'Model': 'K Nearest Neighbour',
#         'Parameters': str(best_knn),
#         'Accuracy': accuracy_score(y_valid, y_pred),
#         'Mean Score': mean_score,
#         'Std Score': std_score
#     })


#NEURAL NETWORKS
#MLP
# param_grid = {
#     'hidden_layer_sizes': [(50,)],
#     'activation': ['logistic'],
#     'alpha': [0.0001],
#     'solver': ['adam'],
#     'learning_rate': ['constant'],
#     'batch_size': [32],
#     'max_iter': [500]
# }
# param_grid = {
#     'hidden_layer_sizes': [(50,), (100,)],
#     'activation': ['logistic', 'relu'],
#     'alpha': [0.0001, 0.001, 0.01],
#     'solver': ['adam', 'sgd'],
#     'learning_rate': ['constant', 'adaptive'],
#     'batch_size': [32, 64, 128],
#     'max_iter': [500]
# }

# mlp = MLPClassifier()
# grid_search = GridSearchCV(mlp, param_grid, cv=kf)
# grid_search.fit(X_train, y_train)
# best_mlp = grid_search.best_estimator_
# y_pred = best_mlp.predict(X_valid)

# # mlp.fit(X_train, y_train)
# # y_pred = mlp.predict(X_valid)
# print("Multi Layer Perceptron")
# print(classification_report(y_valid, y_pred))
# print("Accuracy: ", accuracy_score(y_valid, y_pred))

# scores = cross_val_score(mlp, X, y, cv=kf)
# mean_score = np.mean(scores)
# std_score = np.std(scores)

# print(f"Mean accuracy score: {mean_score:.2f}")
# print(f"Standard deviation: {std_score:.2f}")
# print("************************")

# with open('results.csv', 'a', newline='') as csvfile:
#     fieldnames = ['Model', 'Parameters', 'Accuracy', 'Mean Score', 'Std Score']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow({
#         'Model': 'MLP',
#         'Parameters': str(best_mlp),
#         'Accuracy': accuracy_score(y_valid, y_pred),
#         'Mean Score': mean_score,
#         'Std Score': std_score
#     })
# joblib.dump(best_mlp, 'mlp_best_params.pkl')


#feature ranking and pruning via sensitivity analysis


#running each fake OCR detection model 10 times and gathering the average
#performance of the 10 models


#archive

# def real_or_fake(label):
#     if label == '__label2__':
#         return 'real'
#     else:
#         return 'fake'






