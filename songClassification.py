# NOTE: This was originally a Jupyter Notebook and we
# have not actually run this script because it would take
# a while, so there might be conflicts, but each section should 
# run by itself after imports and defining functions.

# We tried to make sure that all the variable references are correct 
# in this file, but since we got most of our output results from
# different jupyter−notebook sessions there could be some errors in 
# here. We are however confident that the outputs listed in here as 
# comments are results from using correct variable references,
# and the names of the variables do reflect our intent.

import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl_pre
import sklearn.neighbors as skl_nb
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
from sklearn.ensemble import AdaBoostClassifier 
from sklearn import tree

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# User warnings about collinear variables , we think
# the brute forcing approach bypasses this problem. 


################# 
### Functions ### 
#################

# Cross validation with shuffling of data.
# Takes model and fits data. x_columns is an array
# of strings and y_column is a string. n is the desired 
# amount of folds. Returns mean score.
# We are aware of the built-in function but chose to
# define our own for learning purposes.
def crossValidate (model, data, x_columns, y_column, n):
    fold_size = np.ceil(data.shape[0]/n)
    randomized_indices = np.random.choice(
                                            data.shape [0], 
                                            data.shape [0], 
                                            replace=False
                                            )
    all_scores = []
    actual_folds = int(np.ceil(data.shape[0] / fold_size)) 
    for i in range(actual_folds):
        validation_index = np.arange(
            i*fold_size,
            min(i*fold_size+fold_size,
                data.shape[0]),
                1
                ).astype('int') 
    
        current_fold = randomized_indices[validation_index]
        train = data.iloc[~data.index.isin(current_fold)] 
        validation = data.iloc[current_fold]
    
        x_train = train[x_columns]
        y_train = train[y_column] 
        x_validation = validation[x_columns] 
        y_validation = validation[y_column]
    
        model.fit(x_train, y_train) 
        all_scores.append(model.score(x_validation , y_validation))
    return np.mean(all_scores)


#Leave one out cross validation , so same as above without the n. 

def leaveOneOut(model , data , x_columns , y_column ):
    all_scores = []
    for i in range(data.shape[0]):
        validation = data.iloc[i]
        train = data.iloc[~data.index.isin([i])]
        
        x_train = train [x_columns]
        y_train = train [y_column]
        x_validation = validation[x_columns].values.reshape(1, -1) 
        y_validation = validation [y_column].reshape (1 , -1)
         
        model.fit(x_train, y_train) 
        all_scores.append(model.score(x_validation , y_validation))
    return np.mean(all_scores)


#Brute forces combinations of input variables from x_columns and
# returns the combnation that resulted in the best score. n and m
# specifies minimum and maximum amount of input variables to consider 
# when creating the combinations. Uses crossValidate to compare
# different combinations , cross_folds determines the amount of folds. 

def findBestX(model, data, x_columns, y_column, n, m, cross_folds):
    result = (0, [])
    for number_to_choose in range (n , m+1):
        X_combinations = itertools.combinations(
                                                x_columns, 
                                                number_to_choose)
        for X_tup in X_combinations:
                X=[ x for x in X_tup]
                score = crossValidate( model,
                                      data, 
                                      X,
                                      y_column, 
                                      cross_folds)
                if score > result [0]: 
                    result = (score, X)
    return result



####################
### Reading data ###
####################
data = pd.read_csv('training_data.csv')

#data_nodummies should always be the data without dummy variables 
data_nodummies = data
data_dummies = pd.get_dummies(data, columns=['time_signature'])
x_columns = data_dummies.columns.drop(['label']) 
x_columns_nodummies = data_nodummies.columns.drop([ 'label']) 
y_column = 'label'

#Also scales all the data points to 0−1 for use in knn 
skl_minmax = skl_pre.MinMaxScaler()
data_dummies_scaled = pd.DataFrame(
    skl_minmax.fit_transform(data_dummies), 
    columns=data_dummies.columns
    )

#data should be data_dummies , except for KNN
# when it instead should be data_dummies_scaled. 

data = data_dummies

#Creating train and test sets for some initial testing.
trainI = np.random.choice(data.shape[0], size = 375, replace = False) 
trainIndex = data.index.isin(trainI)
train = data.iloc[trainIndex]
test = data.iloc[~trainIndex]

X_train = train[x_columns]
Y_train = train[y_column] 
X_test = test[x_columns]
Y_test = test[y_column]


########### 
### KNN ###
###########

#Using scaled data for KNN 
data = data_dummies_scaled

#Examines an appropriate range of k’s to test for two
# different sets of input variables that proved interesting
# during initial testing. From the plots one can conclude
# that k’s between 3 and 51 seems to be the most interesting. 
k_range = range (1 , 200)

scores = []
X = ['speechiness', 'loudness']
for k in k_range:
    model = skl_nb.KNeighborsClassifier( n_neighbors = k)
    result = crossValidate (model , data , X, y_column , 10)
    scores.append(result) 
plt.plot(k_range, scores)

scores = []
X = ['acousticness', 'danceability', 'energy', 'loudness',
'speechiness' , 'tempo'] 

for k in k_range:
    model = skl_nb.KNeighborsClassifier( n_neighbors = k)
    result = crossValidate (model , data , X, y_column , 10)
    scores.append(result) 
plt.plot(k_range, scores)

#Finds out which combinations of the input variables that usually 
# gets good results. Only tests for odd k’s since sklearn’s knn
# doesn’t handle ties very well. Also only tests for a maximum of 
# 6 input variables since this part takes a while (hours) to run.

first = time.time()
MIN_LENGTH = 1
#Higher MAX_LENGTH would be fun, but it takes too long.
MAX_LENGTH = 6 
CROSS_FOLDS = 10
K_RANGE = range(3, 52, 2)

win_counter = {} 
score_counter = {} 
for k in K_RANGE:
    model = skl_nb.KNeighborsClassifier( n_neighbors = k) 
    result = findBestX(
        model, 
        data, 
        x_columns, 
        y_column,
        MIN_LENGTH, 
        MAX_LENGTH, 
        CROSS_FOLDS
    )
    win_counter[str(result[1])] = win_counter.get(
        str(result[1]), 
        0) + 1 
    score_counter[str(result[1])] = score_counter.get(
        str(result[1]), 
        0) + result [0]

winner = ''
wins = 0
score = 0
for key in win_counter.keys():
    if win_counter[key] > wins:
        wins = win_counter[key]
        winner = key
        score = score_counter [key] / wins
        
#Winner is the combination that wins for many different k’s, 
# not necessarily the one with the highest score overall.
print(winner, score) 
then = time.time ()

#Saves results in a file for better overview. 
with open('results.txt', 'w') as f:
    f.write('time: ' + str(then-first) + 'seconds\n')
    f.write('wins \twinner \t \t \t \t \t \t \t \t \t \tscore \n') 
    for key , value in sorted(
        win_counter.items(),
        key = lambda winner: winner[1], 
        reverse=True):
        line = str(value) + '\t' + str(key) + '\t\t'
        line = line + str(score_counter[key] / value) + '\n' 
        f.write(line)
    f.close()
    
#The results indicates that the following input variables might
# yield the highest scores, this parts tests for the definite best 
# combination and choice of k.

X_to_test = [] 
X_to_test.append(
    ['acousticness' , 'danceability', 'energy', 'loudness' , 'speechiness' , 'tempo']
    )
X_to_test.append(
    ['acousticness', 'danceability', 'duration', 'loudness' , 'speechiness' , 'tempo']
    )
X_to_test.append(
    ['acousticness', 'danceability', 'instrumentalness', 'loudness' , 'speechiness' , 'tempo']
    )
X_to_test.append(
    ['acousticness', 'danceability', 'loudness' , 'speechiness' , 'tempo', 'time_signature_1']
)

result = (0, 0, [])
for k in range (3 , 51):
    model = skl_nb.KNeighborsClassifier() n_neighbors = k) 
    for X in X_to_test:
        score = leaveOneOut(model,
                            data,
                            X, 
                            y_column)
        if score > result [0]:
            result = (score , k, X) 
print(result)

#sample output :
#(0.8413333333333334 , 3 ,
# [’acousticness ’, ’danceability ’, ’energy’, 
# ’loudness ’ , ’speechiness ’ , ’tempo ’])

#Since k=3 leads to suspicions of overfitting, the same test 
# was conducted with k > 3.
# Best result should be the one above, but since k = 3 might 
# be a bit overfit this test is for comparison.

result = (0, 0, [])
for k in range (5 , 16):
    model = skl_nb.KNeighborsClassifier( n_neighbors = k) 
    for X in X_to_test:
        score = leaveOneOut(model,
                            data, 
                            X, 
                            y_column)
        if score > result [0]:
            result = (score , k, X) 
print(result)

#sample output :
#(0.8373333333333334 , 7 ,
# [’acousticness ’, ’danceability ’, ’energy’, 
# ’loudness ’ , ’speechiness ’ , ’tempo ’])



########################### 
### Logistic Regression ### 
###########################

model = skl_lm.LogisticRegression()
data = data_dummies #Switching back to data_dummies

#Testing with a train and test set and fitting the model using 
# all input variables.

model.fit(X_train , Y_train)
predict_prob = model.predict_proba(X_test)
prediction = np.empty(len(X_test), dtype = object) 
prediction = np.where(predict_prob[:, 0] >= 0.5, 0, 1)
print(pd.crosstab(prediction , Y_test)) 

#sample output :
#  0  1
#0 69 48
#1 76 182

print (np.mean( prediction == Y_test ))
#sample output : 
#0.6693333333333333

#Brute force the best choice of input variables MIN_LENGTH = 1
MAX_LENGTH = 16
CROSS_FOLDS = 10
result = findBestX(model,
                   data, 
                   x_columns, 
                   y_column,
                   MIN_LENGTH, 
                   MAX_LENGTH, 
                   CROSS_FOLDS) 
print(result)

#sample output :
#(0.8173333333333334 ,
# [’acousticness ’, ’danceability ’, ’instrumentalness ’,
# ’liveness ’ , ’loudness ’ , ’speechiness ’ , ’time_signature_3 ’ , 
# ’time_signature_5 ’])

#Comparing with the train/test results from earlier
X = ['acousticness', 'danceability', 'instrumentalness',
     'liveness' , 'loudness' , 'speechiness' ,
     'time_signature_3' , 'time_signature_5']
X_train_logreg = train[X]
X_test_logreg = test[X]
model.fit(X_train_logreg , Y_train)
predict_prob = model.predict_proba(X_test_logreg) 
prediction = np.empty(len(X_test_logreg), dtype = object) 
prediction = np.where(predict_prob[:, 0] >= 0.5, 0, 1) 
print(pd.crosstab(prediction , Y_test))

#sample output : 
#   0   1
#0 115 41
#1 30 189
print (np.mean(prediction== Y_test))

#sample output : 
#0.8106666666666666

#Estimating how the model would perform when all data is used 
# for training
score = leaveOneOut(model, data, X, y_column)
print(score) #0.8066666666666666

#With Logistic Regression, we also tested if it would
# perform better if time_signature was not treated
# as qualitative, with no dummy variables.
#The following X was found with findBestX for an x_columns 
# without dummy variables :
X = ['acousticness', 'danceability', 'energy', 
     'instrumentalness' , 'loudness','speechiness', 
     'time_signature', 'valence']
score = leaveOneOut(model , data_nodummies , X, y_column) 
print(score) #0.8106666666666666

#We also created ROC diagrams for a model using all predictor 
# variables and for a model
# using the predictor variables in X ( just above ).
# This was done without the dummy variables
# as Logistic Regression performed better without them. 
# The ROC diagrams are displayed in the report.

#This code was used to generate the diagrams, being run twice
# with X_test being "test_nodummies[x_columns]"
# and then "test_nodummies[X]". Unfortunately we lost the exact 
# code and we thus left this part commented.
####
#false_positive_rate = []
#true_postive_rate = []
#N = np.sum(Y_test == 0)
#P = np.sum(Y_test == 1)
#threshold = np.linspace(0.01, 0.99, 99)
#model = skl_lm.LogisticRegression(solver = ’liblinear ’) 
#model.fit (X_test , Y_test)
#predict_prob = model.predict_proba(X_test)
#for i in range(len(threshold )):
# prediction = np.empty(len(X_test), dtype=object)
#prediction = np.where(predict_prob[:, 0] > threshold[i], 0, 1)
#FP = np.sum((prediction == 1)&(Y_test == 0)) 
#TP = np.sum((prediction == 1)&(Y_test == 1))
#false_positive_rate.append(FP/N) 
# true_postive_rate .append(TP/P)
#plt.plot(false_positive_rate , true_postive_rate) 
####


################# 
### LDA & QDA ### 
#################
data = data_dummies
LDAmodel = skl_da.LinearDiscriminantAnalysis() 
QDAmodel = skl_da.QuadraticDiscriminantAnalysis()

#Some initial testing with the train / test data set 
LDAmodel . f i t ( X_train , Y_train )
QDAmodel. fit (X_train , Y_train)
print('LDA score: ', LDAmodel.score(X_test ,Y_test)) 
print ('QDA score : ' , QDAmodel.score(X_test , Y_test))

#sample output :
#LDA score: 0.832
#QDA score : 0.4533333333333333
#Testing LDA with all predictor variables since LDA was
# unreasonably good this one time
score = leaveOneOut (LDAmodel , data , x_columns , y_column )
print(score) 
#0.804

#Finding best predictor variables for LDA from 1 variable to 6 MIN_LENGTH = 1
MAX_LENGTH = 6
CROSS_FOLDS = 10
result = findBestX(LDAmodel,
                   data, 
                   x_columns, 
                   y_column,
                   MIN_LENGTH, 
                   MAX_LENGTH, 
                   CROSS_FOLDS
                   ) 
print(result)
#sample output :
#(0.8280000000000001 ,
# [’acousticness ’, ’danceability ’, ’instrumentalness ’, 
# ’liveness ’ , ’loudness ’ , ’speechiness ’])
#leaveOneOut score for the suggested predictor variables above 

X = ['acousticness', 'danceability', 'instrumentalness',
     'liveness', 'loudness', 'speechiness']
score = leaveOneOut(LDAmodel, data, X, y_column) 
print(score)
#0.8253333333333334

#We also tried finding the best predictor variables in 
# a different range for LDA

MIN_LENGTH = 8
MAX_LENGTH = 16 
CROSS_FOLDS = 10
result = findBestX(LDAmodel,
                   data, 
                   x_columns, 
                   y_column,
                   MIN_LENGTH, 
                   MAX_LENGTH, 
                   CROSS_FOLDS
                   ) 
print(result)
#sample output: #(0.8293333333333335 ,
# [’acousticness ’, ’duration ’, ’instrumentalness ’,
# ’loudness ’, ’speechiness ’, ’tempo’, ’valence ’,
# ’time_signature_1 ’ , ’time_signature_3 ’ , ’time_signature_5 ’])

X = ['acousticness', 'duration', 'instrumentalness',
     'loudness' , 'speechiness', 'tempo', 'valence ', 
     'time_signature_1' , 'time_signature_3' , 'time_signature_5']
score = leaveOneOut(LDAmodel , data , X, y_column ) 
print(score)
#0.8226666666666667

#For QDA we had min and max MIN_LENGTH = 8
MAX_LENGTH = 16
CROSS_FOLDS = 10
result = findBestX(QDAmodel,
                   data, 
                   x_columns, 
                   y_column,
                   MIN_LENGTH , 
                   MAX_LENGTH, 
                   CROSS_FOLDS
                   ) 
#sample output :
length set to 8 and 16 for our first run
#(0.8146666666666667 ,
# [’acousticness ’, ’danceability ’, ’duration ’,
# ’instrumentalness ’ , ’loudness ’ , ’speechiness ’ , # ’tempo’, ’valence ’, ’time_signature_5 ’])
X = ['acousticness', 'danceability', 'duration', 
     'instrumentalness', 'loudness' , 'speechiness' , 
     'tempo', 'valence', 'time_signature_5']
score = leaveOneOut(QDAmodel, data , X, y_column) 
print(score)
#0.7973333333333333

#An overview of how much our choice of predictor variables 
# for LDA and QDA respectively performs
# compared to using all predictor variables
X_l = ['acousticness', 'danceability', 'instrumentalness',
       'liveness', 'loudness', 'speechiness']
X_q = ['acousticness', 'danceability', 'duration', 
       'instrumentalness', 'loudness', 'speechiness',
       'tempo', 'valence', 'time_signature_5'] 
y_column = 'label'

print(
    'LDA with good:', 
    leaveOneOut(LDAmodel, data_dummies, X_l, y_column)
    )

print(
    'LDA with all:',
    leaveOneOut(LDAmodel, data_dummies, x_columns, y_column)
    ) 

print(
    'QDA with good:',
    leaveOneOut(QDAmodel, data_dummies , X_q, y_column)
    )

print(
    'QDA with all:',
    leaveOneOut(QDAmodel, data_dummies , x_columns , y_column) 
    )

#LDA with good : 0.8253333333333334 
#LDA with all : 0.804
#QDA with good : 0.7973333333333333 
#QDA with all : 0.62



################ 
### Boosting ### 
################

#Our initial testing found that the boosting
# classifier performed well with the following hyperparameters.
model = AdaBoostClassifier(
    tree.DecisionTreeClassifier(max_depth=1),
    n_estimators = 180, 
    learning_rate = 0.2
    )
leaveOneOut(model, data , x_columns , y_column )
score = print(score) #0.8173333333333334

# We proceeded to loop over a range of different values for the 
# hyperparameters, with a higher amount of folds for the
# cross validation.
result=[0,0,0]
for n in range (20 ,200 ,10):
    for l in range (1 ,5):
        model = AdaBoostClassifier(
            tree.DecisionTreeClassifier(max_depth=1), 
            n_estimators = n,
            learning_rate=l /10
            )
        score = crossValidate (model, 
                               data,
                               x_columns,
                               y_column, 
                               50)
        if result[0]<score:
            result = [score, n, l/10] 
print(result)
#sample output : 
#[0.8280000000000001, 180, 0.1]

# Based on our output above we then evaluated this using leaveOneOut
model = AdaBoostClassifier(
    tree.DecisionTreeClassifier (max_depth=1),
    n_estimators = 180, learning_rate =0.1
    )
score = leaveOneOut ( model , data , x_columns , y_column )
print(score) #0.8266666666666667