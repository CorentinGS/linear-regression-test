import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model, metrics


def main():
    print("hello world")

    dataset = pd.read_csv("dt2.csv")
    dataset.head()

    # sns.histplot(dataset["Quality"])

    # plt.show()



    x = dataset[['EF', 'Interval', 'Rep', 'CID', 'UID']]
    y = dataset['Quality']

    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size = 0.2, random_state = 100)

    mlr = linear_model.LinearRegression()  
    mlr.fit(x_train, y_train)

    print("Intercept: ", mlr.intercept_)
    print("Coefficients:")
    print(list(zip(x_, mlr.coef_)))

    y_pred_mlr= mlr.predict(x_test)
    x_pred_mlr= mlr.predict(x_train)  


    # print("Prediction for test set: {}".format(y_pred_mlr))

    mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
    print(mlr_diff)

    print('R squared value of the model: {:.2f}'.format(mlr.score(x_,y)*100))

    meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
    meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
    # rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

    print('Mean Absolute Error:', meanAbErr)
    print('Mean Square Error:', meanSqErr)
    # print('Root Mean Square Error:', rootMeanSqErr)


main()