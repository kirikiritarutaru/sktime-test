import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_arrow_head, load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator
from pprint import pprint


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


def univariate_TSC():
    data, labels = load_arrow_head(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(data, labels)
    check_data(x_train, y_train)
    classifier = TimeSeriesForestClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(accuracy_score(y_test, y_pred))


def multivariate_TSC():
    df_train, df_labels = load_basic_motions(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        df_train, df_labels, random_state=42
    )
    print('x_train.shape, y_train.shape, x_test.shape, y_test.shape:')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # pprint(x_train.head())
    print(np.unique(y_train))

    # 多変量の時系列を一つの時系列にまとめて（単変量の時系列にして）、classifierにかける
    # これでもうまくいくんか…
    steps = [
        ('concatenate', ColumnConcatenator()),
        ('classify', TimeSeriesForestClassifier(n_estimators=100))
    ]
    clf = Pipeline(steps)
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))

    # 列ごとに別のclassifierで学習、予測をすることができる。
    clf = ColumnEnsembleClassifier(
        estimators=[
            ('TSF0', TimeSeriesForestClassifier(n_estimators=100), [0]),
            ('BOSSEnsemble3', BOSSEnsemble(max_ensemble_size=5), [3])
        ]
    )
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))


if __name__ == '__main__':
    multivariate_TSC()
