import importlib
import json
import logging
from dataclasses import field
from datetime import datetime
from pathlib import Path

import pandas as pd
import tensorflow as tf
from pydantic.dataclasses import dataclass

logger = logging.getLogger('harnet')


@dataclass
class HARNetCfg:
    # Model
    model: str = "HARNet"
    filters_dconv: int = 1
    use_bias_dconv: bool = False
    activation_dconv: str = "relu"
    lags: list[int] = field(default_factory=lambda: [1, 5, 20])

    # Optimization
    learning_rate: float = 0.0001
    epochs: int = 10000
    steps_per_epoch: int = 1
    label_length: int = 5
    batch_size: int = 4
    optimizer: str = "Adam"
    loss: str = "QLIKE"
    verbose: int = 1
    baseline_fit: str = "WLS"

    # Data
    path_MAN: str = "~/develop/vola/data/MAN_data.csv"
    asset: str = ".SPX"
    include_sv: bool = False
    start_year_train: int = 2012
    n_years_train: int = 4
    start_year_test: int = 2016
    n_years_test: int = 1

    # Preprocessing
    scaler: str = "MinMax"
    scaler_min: float = 0.0
    scaler_max: float = 0.001

    # Save Paths
    tb_path: str = "./tb/"
    save_path: str = "./results/"
    save_best_weights: bool = False

    # Misc
    run_eagerly: bool = False


def year_range_to_idx_range(ts, year_range):
    idx_start = 0
    for k in range(len(ts)):
        if ts.index[k].year == year_range[0]:
            idx_start = k
            break

    while ts.index[k].year < year_range[1]:
        k = k + 1

    return [idx_start, k]


def get_MAN_data(path, asset=".SPX", include_sv=False):  # , estimator="rv5_ss"
    # load the data
    # e.g. asset = .SPX, estimator = rv5_ss (see https://realized.oxford-man.ox.ac.uk/documentation/estimators)
    # returns the specified time series (e.g. realized volatility for the SP500)
    data = pd.read_csv(Path(path))
    data_asset = data[data.Symbol == asset]
    if include_sv:
        ts = data_asset.set_index("date")[['rv5_ss', 'rsv_ss']]
        ts = ts.assign(rsv_pos=(ts['rv5_ss'] - ts['rsv_ss']).values)
        signed_jumps = (ts['rsv_pos'] - ts['rsv_ss']).values
        ts = ts.assign(signed_jumps=signed_jumps)
        del ts['rsv_pos']
        ts.index = pd.DatetimeIndex(pd.to_datetime(ts.index, utc=True))
    else:
        ts = data_asset.set_index("date")[['rv5_ss']]
        ts.index = pd.DatetimeIndex(pd.to_datetime(ts.index, utc=True))

    return ts
