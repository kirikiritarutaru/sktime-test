import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tbats import TBATS
from sktime.performance_metrics.forecasting import \
    mean_absolute_percentage_error as mape_loss
from sktime.utils.plotting import plot_series

# 参考
# https://www.salesanalytics.co.jp/datascience/datascience010/


def check_data(
        y_train: pd.Series,
        y_test: pd.Series = None,
        y_pred: pd.Series = None
):
    if y_test is None and y_pred is None:
        plot_series(y)

    if y_test is not None and y_pred is None:
        plot_series(y_train, y_test, labels=['y_train', 'y_test'])

    if y_test is not None and y_pred is not None:
        plot_series(
            y_train, y_test, y_pred,
            labels=['y_train', 'y_test', 'y_pred']
        )

    plt.show()


def apply_arima(y_train: pd.Series, y_test: pd.Series):
    forecaster = AutoARIMA(sp=12, suppress_warnings=True)
    forecaster.fit(y_train)

    fh = np.arange(len(y_test))+1
    y_pred = forecaster.predict(fh)

    print(mape_loss(y_test, y_pred))
    check_data(y_train, y_test, y_pred)


if __name__ == '__main__':
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=36)
    apply_arima(y_train, y_test)
