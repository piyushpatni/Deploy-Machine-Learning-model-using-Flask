import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
import pickle

states_and_ohe={1:[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       2:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
       3:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
       4: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
       5: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       6:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
       7:[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       8:[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       9: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
       10: [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       11: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
       12: [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       13: [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       14: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
       15:[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
       16: [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       17:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
       18: [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       19: [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
       20: [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       21: [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       22: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
       23: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
       24: [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       25: [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
#print(states_and_ohe[10])

data=pd.read_csv(r"dataset.csv")
import numpy as np
data=data.fillna(np.mean(data))
new_data=data.drop(['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'], axis=1)
features = pd.get_dummies(new_data)

# JAN-FEB TRAINNING
def Jan_Feb_Season(rain,state):
    # Use numpy to convert to arrays
    import numpy as np
    # Labels are the values we want to predict
    labels = np.array(features['ANNUAL'])
    final_features= features.drop(features.columns[[0,2,3,4]], axis = 1)
    # Convert to numpy array
    finall_features = np.array(final_features)
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(finall_features, labels, test_size = 0.25, random_state = 42)
    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    for states, ohe in states_and_ohe.items():
        if state==states:
            s=ohe
    y=[]
    y.append(rain+s)
    #print(y)
    predicted_data=[]
    predicted_data= rf.predict(y)
    return predicted_data


pickle.dump(Jan_Feb_Season,open("model1.pkl",'wb'),pickle.HIGHEST_PROTOCOL)
model1=pickle.load(open('model1.pkl','rb'))
#print(model1([1143.900000],1))

# Mar-May TARINNING
def Mar_May_Season(rain,state):
    # Use numpy to convert to arrays
    import numpy as np
    # Labels are the values we want to predict
    labels_1 = np.array(features['ANNUAL'])
    final_features_1= features.drop(features.columns[[0,1,3,4]], axis = 1)
    # Convert to numpy array
    finall_features_1 = np.array(final_features_1)
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(finall_features_1, labels_1, test_size = 0.25, random_state = 42)
    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    for states, ohe in states_and_ohe.items():
        if state==states:
            s=ohe
    y=[]
    y.append(rain+s)
    predicted_data=[]
    predicted_data= rf.predict(y)
    return predicted_data

pickle.dump(Mar_May_Season,open("model2.pkl",'wb'),pickle.HIGHEST_PROTOCOL)
model2=pickle.load(open('model2.pkl','rb'))
#print(model2([1143.900000],1))

# Jun-Sep TRAINNING
def Jun_Sep_Season(rain,state):
    # Use numpy to convert to arrays
    import numpy as np
    # Labels are the values we want to predict
    labels_2 = np.array(features['ANNUAL'])
    final_features_2= features.drop(features.columns[[0,1,2,4]], axis = 1)
    # Convert to numpy array
    finall_features_2 = np.array(final_features_2)
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(finall_features_2, labels_2, test_size = 0.25, random_state = 42)
    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    for states, ohe in states_and_ohe.items():
        if state==states:
            s=ohe
    y=[]
    y.append(rain+s)
    predicted_data=[]
    predicted_data= rf.predict(y)
    return predicted_data

pickle.dump(Jun_Sep_Season,open("model3.pkl",'wb'),pickle.HIGHEST_PROTOCOL)
model3=pickle.load(open('model3.pkl','rb'))
#print(model3([1143.900000],1))

# Oct-Dec TRAINNING
def Oct_Dec_Season(rain,state):
    # Use numpy to convert to arrays
    import numpy as np
    # Labels are the values we want to predict
    labels_3 = np.array(features['ANNUAL'])
    final_features_3= features.drop(features.columns[[0,2,3,4]], axis = 1)
    # Convert to numpy array
    finall_features_3 = np.array(final_features_3)
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(finall_features_3, labels_3, test_size = 0.25, random_state = 42)
    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    for states, ohe in states_and_ohe.items():
        if state==states:
            s=ohe
    y=[]
    y.append(rain+s)
    predicted_data=[]
    predicted_data= rf.predict(y)
    return predicted_data

pickle.dump(Oct_Dec_Season,open("model4.pkl",'wb'),pickle.HIGHEST_PROTOCOL)
model4=pickle.load(open('model4.pkl','rb'))
#print(model4([1143.900000],1))

print("Done")
