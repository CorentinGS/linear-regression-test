# Compare Algorithms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def comp():
    # prepare data
    dataset = pd.read_csv("data3.csv")
    dataset.head()

    #dataset["quality"][dataset["quality"] < 4] = 0
    #dataset["quality"][dataset["quality"] >= 4] = 1

    x = dataset[["user_id", "card_id", "repetition", "efactor", 'interval']]
    y = dataset["quality"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # prepare models
    models = [
        ('CART', DecisionTreeClassifier()), ('RFC', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('XGBC', XGBClassifier(n_estimators=200, max_depth=20, learning_rate=1.5, use_label_encoder=False)),
        ("LGMC", LGBMClassifier(random_state=42))
    ]
    # evaluate each model in turn
    results = [[], [], []]
    names = []

    i = 0
    for name, model in models:
        print(f"==============={name}================")
        model.fit(x_train, y_train)

        names.append(name)

        y_pred_mlr = model.predict(x_test)

        mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})

        r = model.score(x, y)

        print('R squared value of the model: {:.2f}'.format(r * 100))

        meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
        meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
        rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

        print('Mean Absolute Error:', meanAbErr)
        print('Mean Square Error:', meanSqErr)
        print('Root Mean Square Error:', rootMeanSqErr)
   

        # toto = model.predict(([[6, 280, 5, 2, 2.9727996826171875]]))
        # print(toto)

        results[0].append(i)
        results[1].append(r)
        results[2].append(meanAbErr)
        i += 1


        print('Classification Report:\n', classification_report(y_test, y_pred_mlr))
        print('Accuracy Score:', accuracy_score(y_test, y_pred_mlr) * 100)

    # boxplot algorithm comparison
    fig, ax = plt.subplots()
    fig.suptitle('Algorithm Comparison')
    ax.errorbar(results[0], results[1], results[2], fmt='o', linewidth=2, capsize=6)
    plt.show()


def main():
    comp()


main()
