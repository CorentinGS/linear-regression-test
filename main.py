import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics


def main():
    print("hello world")

    dataset = pd.read_csv("dt3.csv")
    dataset.head()

    # sns.histplot(dataset["Quality"])

    # plt.show()

    x = dataset[['efactor', 'interval', 'repetition', 'CID', 'UID']]
    y = dataset['quality']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

    mlr = linear_model.LinearRegression()  
    mlr.fit(x_train, y_train)

    print("Intercept: ", mlr.intercept_)
    print("Coefficients:")
    print(list(zip(x, mlr.coef_)))

    y_pred_mlr= mlr.predict(x_test)
    x_pred_mlr= mlr.predict(x_train)  


    # print("Prediction for test set: {}".format(y_pred_mlr))

    mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
    print(mlr_diff)

    print('R squared value of the model: {:.2f}'.format(mlr.score(x,y)*100))

    meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
    meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
    # rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

    print('Mean Absolute Error:', meanAbErr)
    print('Mean Square Error:', meanSqErr)
    # print('Root Mean Square Error:', rootMeanSqErr)

    toto = mlr.predict(([[1.4567999839782715, 3, 2,18,6]]))

    if toto >= 3:
        print("Good answer")
    else:
        print("Bad answer")

    print(toto)

main()