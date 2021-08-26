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


def plot_between(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    pred_ints: pd.DataFrame,
    alpha: float = 0.05
):
    fig, ax = plot_series(
        y_train,
        y_test,
        y_pred,
        labels=['y_train', 'y_test', 'y_pred']
    )
    ax.fill_between(
        ax.get_lines()[-1].get_xdata(),
        pred_ints['lower'],
        pred_ints['upper'],
        alpha=0.2,
        color=ax.get_lines()[-1].get_c(),
        label=f'{100*(1-alpha)}% prediction intervals'
    )
    ax.legend()
    plt.show()


# ARIMA モデル
def apply_arima(y_train: pd.Series, y_test: pd.Series):
    forecaster = AutoARIMA(sp=12, suppress_warnings=True)
    forecaster.fit(y_train)

    fh = np.arange(len(y_test))+1
    y_pred, pred_ints = forecaster.predict(
        fh, return_pred_int=True, alpha=0.05
    )

    print(type(pred_ints))
    print(mape_loss(y_test, y_pred))
    check_data(y_train, y_test, y_pred)
    plot_between(y_train, y_test, y_pred, pred_ints)


# 指数平滑化法
def apply_ES(y_train: pd.Series, y_test: pd.Series):
    forecaster = ExponentialSmoothing(
        trend='add', seasonal='multiplicative', sp=12
    )
    model = forecaster.fit(y_train)
    fh = np.arange(len(y_test))+1
    y_pred = forecaster.predict(fh)
    print(mape_loss(y_test, y_pred))
    plot_series(
        y_train, y_test, y_pred,
        labels=['y_train', 'y_test', 'y_pred']
    )
    plt.show()


# Error-Trend-Seasonality （状態空間指数平滑化法）
def apply_ETS(y_train: pd.Series, y_test: pd.Series):
    forecaster = AutoETS(auto=True, sp=12, n_jobs=-1)
    fh = np.arange(len(y_test))+1
    y_pred = forecaster.predict(fh)
    print(mape_loss(y_test, y_pred))
    plot_series(
        y_train, y_test, y_pred,
        labels=['y_train', 'y_test', 'y_pred']
    )
    plt.show()


# TBATS
# ( Exponential smoothing state space model with Box-Cox transformation,
# ARMA errors, Trend and Seasonal components )
def apply_TBATS(y_train: pd.Series, y_test: pd.Series):
    forecaster = TBATS(sp=12, use_trend=True, use_box_cox=True)
    forecaster.fit(y_train)
    fh = np.arange(len(y_test))+1
    y_pred = forecaster.predict(fh)
    print(mape_loss(y_test, y_pred))
    plot_series(
        y_train, y_test, y_pred,
        labels=['y_train', 'y_test', 'y_pred']
    )
    plt.show()


def apply_ensemble(y_train: pd.Series, y_test: pd.Series):
    forecaster = EnsembleForecaster(
        [
            ('ARIMA', AutoARIMA(sp=12, suppress_warnings=True)),
            ('ES',
                ExponentialSmoothing(
                    trend='add', seasonal='multiplicative', sp=12
                )),
            ('ETS', AutoETS(auto=True, sp=12, n_jobs=-1)),
            ('TBATS', TBATS(sp=12, use_trend=True, use_box_cox=False))
        ]
    )
    forecaster.fit(y_train)
    fh = np.arange(len(y_test))+1
    y_pred = forecaster.predict(fh)
    print(mape_loss(y_test, y_pred))
    plot_series(
        y_train, y_test, y_pred,
        labels=['y_train', 'y_test', 'y_pred']
    )
    plt.show()


if __name__ == '__main__':
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=36)

    # apply_arima(y_train, y_test)
    # apply_ES(y_train, y_test)
    # apply_ETS(y_train, y_test)
    # apply_TBATS(y_train, y_test)
    apply_ensemble(y_train, y_test)
