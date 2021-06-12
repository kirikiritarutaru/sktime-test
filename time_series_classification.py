from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_arrow_head, load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor


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


# Time Series Forest Classifier: 間隔ベース (interval-based) の手法
# 学習：
# 1. ランダムな開始位置と長さで、時系列データを分割
# 2. 各区間から特徴量（平均、標準偏差、勾配）を抽出
# 3. 2で抽出した特徴量から、その区間の決定木を学習
# 4. 決定木を複数作成
# 予測：
# 決定木のアンサンブルとって、予測したいデータを分類
def univariate_TSC():
    data, labels = load_arrow_head(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(data, labels)
    check_data(x_train, y_train)
    classifier = TimeSeriesForestClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(accuracy_score(y_test, y_pred))


# BOSS (Bag of SFA Symbols): 辞書ベース (Dictionary-based) の手法
# 1. 長さ w のスライディングウィンドウを時系列データに適用
# 2. ウィンドウ内の時系列データをフーリエ変換
# 3. Multiple Coefficient Binning を使用して、最初の l 個のフーリエ項をシンボルに離散化し、Token を生成
# 4. ウィンドウを動かしながら Token の頻度をカウント、時系列データを Token のヒストグラムに変換
# 5. 時系列データから抽出された Token ヒストグラムで任意の分類器をトレーニング
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


def check_basic_motion():
    df_train, df_labels = load_basic_motions(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        df_train, df_labels, random_state=42
    )
    labels, counts = np.unique(y_train, return_counts=True)
    print(labels, counts)

    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    for label in labels:
        x_train.loc[y_train == label, 'dim_0'].iloc[0].plot(ax=ax, label=label)
    plt.legend()
    ax.set(title='Example time series', xlabel='Time')
    plt.show()

    for label in labels[:2]:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
        for instance in x_train.loc[y_train == label, 'dim_0']:
            ax.plot(instance)
        ax.set(title=f'Instancesof {label}')
    plt.show()


# tsfreshを用いた「時系列からの自動特徴抽出」
def tsfresh():
    df_train, df_labels = load_basic_motions(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        df_train, df_labels, random_state=42
    )
    labels, counts = np.unique(y_train, return_counts=True)

    transformer = TSFreshFeatureExtractor(default_fc_parameters='minimal')
    extracted_features = transformer.fit_transform(x_train)
    print('Extracted features using tsfresh')
    print(extracted_features.head())

    classifier = make_pipeline(
        TSFreshFeatureExtractor(show_warnings=False),
        RandomForestClassifier()
    )
    classifier.fit(x_train, y_train)
    print(f'score: {classifier.score(x_test, y_test)}')


if __name__ == '__main__':
    tsfresh()
