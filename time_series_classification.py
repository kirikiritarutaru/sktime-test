import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
import pandas as pd


def check_data(df_train: pd.DataFrame, df_labels: pd.Series):
    labels, counts = np.unique(df_labels, return_counts=True)
    print(labels, counts)
    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    for label in labels:
        df_train.loc[y_train == label, 'dim_0'].iloc[0].plot(
            ax=ax, label=f'class {label}'
        )
    plt.legend()
    ax.set(title='Example time series', xlabel='Time')
    plt.show()


if __name__ == '__main__':
    data, labels = load_arrow_head(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(data, labels)
    check_data(x_train, y_train)
    classifier = TimeSeriesForestClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(accuracy_score(y_test, y_pred))
