import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, linear_model, metrics


def main():
    print("hello world")

    dataset = pd.read_csv("dt3.csv")
    dataset.head()

    # sns.histplot(dataset["Quality"])

    # plt.show()

    #x_ = dataset[['CID', 'UID']]
    x_ = dataset[['efactor',  'repetition', 'UID', 'CID']]
    y = dataset['interval']

    # x_ = PolynomialFeatures(degree=3, include_bias=False).fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size = 0.3, random_state = 30)

    #dtree = DecisionTreeRegressor(max_depth=10, min_samples_leaf=0.13, random_state=3)
    #dtree.fit(x_train, y_train)

    model_rf = RandomForestRegressor(n_estimators=150,  max_features=1, oob_score=True, random_state=30)
    model_rf.fit(x_train, y_train) 


    y_pred_mlr= model_rf.predict(x_test)


    mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
    print(mlr_diff)

    print('R squared value of the model: {:.2f}'.format(model_rf.score(x_,y)*100))

    meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
    meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

    print('Mean Absolute Error:', meanAbErr)
    print('Mean Square Error:', meanSqErr)
    print('Root Mean Square Error:', rootMeanSqErr)

    toto = model_rf.predict(([[2.58463262, 6, 6, 15]]))
    print(toto)

main()