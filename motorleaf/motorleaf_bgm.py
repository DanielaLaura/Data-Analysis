# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
# Splitting data into training and testing
import random
from sklearn.model_selection import train_test_split
# Machine Learning Models
from sklearn.ensemble import GradientBoostingRegressor

# Load data
data = pd.read_csv('dataset.csv')

# Function that returns mean absolute percentage error
def mape(y_true, y_pred):
    return (np.mean(abs((y_true - y_pred)/y_true)))*100

# function which takes input a dataframe(x) that can contain both numeric and categorical columns
def fixoutliers(x):
    '''
        Objective:
            Fixes outliers in a dataframe  which are not between sett minimum and maximum end
            outliers
        Inputs:
            dataframe

        Output:
            dataframe that contains only the non-extreme outlier features
        '''
    # Get all the column name from the input dataframe x
    xColumnNames = x
    # keep the labels column as we do't want to alter it
    y = xColumnNames['yield']
    xColumnNames.drop('yield', axis=1)
    #print(xColumnNames)
    # for j in df2ColumnNames:

    for j in xColumnNames:
        try:
            #print("colnames ", j)
            xy = x[j]
            mydata = pd.DataFrame()
            # print(xy)
            updated = []
            Q1, Q3 = np.percentile(xy, [25, 75])
            IQR = Q3 - Q1
            minimum = Q1 - 1.5 * IQR
            maximum = Q3 + 1.5 * IQR
            for i in xy:  # alter columns if they are not in within the range
                if (i > maximum):
                    #print("Entering maxim")
                    i = maximum
                    updated.append(i)
                elif (i < minimum):
                    #print("enterinf minimum")
                    i = minimum
                    updated.append(i)
                else:
                    updated.append(i)
            x[j] = updated
        except:
            continue

    x['yield'] = y  # add back the labels column
    return x

#Remove outliers
print('Removing outliers!')
new_data=fixoutliers(data)

#drop the columns with constant value of zero
new_data.drop(['5','315','625','935','1245','1555','1865','2175','2485','2795','3105','3415','3725','4035'], axis=1, inplace=True)

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold.
    Inputs:
        threshold: any features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Dont want to remove correlations between yield|
    y = x['yield']
    x = x.drop(columns=['yield'])

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)

    # Add the score back in to the data
    x['yield'] = y

    return x

# Remove collinear features
print('Removing collinear features!')
features = remove_collinear_features(new_data, 0.75);

#Divide data set into train and test set
def prepare_data(data):
    train = data.iloc[0:115].copy()
    test = data.iloc[115:131].copy()
   # Display sizes of data
    print('Training Feature Size: ', train.shape)
    print('Testing Feature Size:  ', test.shape)
    #keep labes for train  set
    y = train['yield']
    #keep ids for test set
    ids_test = test['week_num']
    # Remove the ids and target
    X = train.drop(columns = ['week_num', 'yield'])
    X = np.array(X)

    # Remove the ids and target
    X_test = test.drop(columns=['week_num', 'yield'])
    X_test = np.array(X_test)

    return X,y, X_test, ids_test


print('Dividing data into train&test set!')
X,y, X_test, ids_test = prepare_data(features)

gradient_boosted = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=2, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=4,
             min_samples_split=6, min_weight_fraction_leaf=0.0,
             n_estimators=200, presort='auto', random_state=42,
             subsample=1.0, verbose=0, warm_start=False)

# Takes in a model, trains the model, and evaluates the model on the validation set
def fit_and_evaluate(model):
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    model.fit(x_train, y_train)
    # Make predictions and evalute
    preds = model.predict(x_valid)
    map_error = mape(y_valid, preds)
    print('MAPE = ' + str(map_error))

    # Return the performance metric
    return map_error

print('Fitting and evaluating model!')
gbm_mape = fit_and_evaluate(gradient_boosted)

#make prediction on the test set
preds = gradient_boosted.predict(X_test)

print('Fetching prediction on test set!')
# Prepare submission
subm = pd.DataFrame()
subm['week_num'] = ids_test.values
subm['yield'] = preds
subm.to_csv('submission_gbm.csv', index=False)