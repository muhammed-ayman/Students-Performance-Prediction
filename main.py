import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv('student-mat.csv', sep=";") # Reading the data with pandas with ; as a separator
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']] # Get only G1, G2, G3, studytime, failures, absences to see their relation with each other

value_to_predict = 'G3' # This is the value that the model will be predicting

x = np.array(data.drop(columns=value_to_predict)) # Here we drop the to-predict column
y = np.array(data[value_to_predict]) # Here we grap the values of the to-predict column for training purposes
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1) # This utilizes sklearn to split the data into training and testing sections with 10% of the data dedicated for the testing one

## Training the model ::
# Uncomment the following lines if you need to get a better trained model
# best_score = 0
# for _ in range(30): # Statistically speaking, you can increase the range value such that you get a higher chance of getting a higher accuracy
#     linear = linear_model.LinearRegression() # Initializing a linear regression model
#     linear.fit(x_train, y_train) # Fitting the training data into the model
#     accuracy = linear.score(x_test, y_test) # Calculating the accuracy of the generated model
#
#     if accuracy > best_score:
#         best_score = accuracy
#         with open('student-model.pickle', 'wb') as file: # Using pickle to save the trained model in case it has a better accuracy than the previous one
#             pickle.dump(linear, file)


pickle_in = open('student-model.pickle', 'rb')
linear = pickle.load(pickle_in) # Loading the last saved model with pickle

predictions = linear.predict(x_test) # Grap the predicted values of the test data and put them into the predictions variable

print('----------------------')
print('Prediction ', 'True Value')
print('----------------------')
for x in range(len(predictions)):
    print(f'{round(predictions[x],2)}{(12-len(str(round(predictions[x],2))))*" "}{round(y_test[x],2)}') # Printing the predicted vs true values

# Drawing a graph between one of the features and the label
feature = 'failures'
style.use('ggplot')
pyplot.scatter(data[feature], data[value_to_predict])
pyplot.xlabel(feature)
pyplot.ylabel(value_to_predict)
pyplot.show()
