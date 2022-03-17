import logging

import joblib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.layers import Conv1D, Dense

logger = logging.getLogger('harnet')

DTYPE = tf.float32


def scaler_from_cfg(cfg):
    if cfg.scaler == "MinMax":
        return MinMaxScaler(cfg.scaler_min, cfg.scaler_max)
    elif cfg.scaler == "Log":
        return LogScaler()
    elif cfg.scaler == "LogMinMax":
        return LogMinMaxScaler(cfg.scaler_min, cfg.scaler_max)
    elif cfg.scaler == "None":
        return NoneScaler()
    else:
        logging.warning(f"Scaler {cfg.scaler} unknown. Using NoScaler.")
        return NoneScaler()


def get_HARSVJ_baseline_fit(lags, baseline_fit, ts_norm, idx_range_train):
    harsvj = HARSVJ(lags, baseline_fit)
    harsvj.fit(ts_norm.values[idx_range_train[0] - harsvj.max_lag:idx_range_train[1], :])
    return harsvj.coeffs


def get_HAR_baseline_fit(lags, baseline_fit, ts_norm, idx_range_train):
    har = HAR(lags, baseline_fit)
    har.fit(ts_norm.values[idx_range_train[0] - har.max_lag:idx_range_train[1], :])
    return har.coeffs


def model_from_cfg(cfg, ts_norm, scaler, idx_range_train):
    clip_value = scaler.transform(
        0.5 * scaler.inverse_transform(np.min(ts_norm.values[idx_range_train[0]:idx_range_train[1], 0])))
    if cfg.model == "HAR":
        model = HAR(cfg.lags, cfg.baseline_fit, clip_value)
    elif cfg.model == "HARSVJ":
        model = HARSVJ(cfg.lags, cfg.baseline_fit, clip_value)
    elif cfg.model == "HARNet":
        regr_coeffs = get_HAR_baseline_fit(cfg.lags, cfg.baseline_fit, ts_norm, idx_range_train)
        model = HARNet(cfg.filters_dconv, cfg.use_bias_dconv, cfg.activation_dconv, cfg.lags, regr_coeffs, clip_value)
    elif cfg.model == "HARNetSVJ":
        regr_coeffs = get_HARSVJ_baseline_fit(cfg.lags, cfg.baseline_fit, ts_norm, idx_range_train)
        model = HARNetSVJ(cfg.filters_dconv, cfg.use_bias_dconv, cfg.activation_dconv, cfg.lags, regr_coeffs,
                          clip_value)
    elif cfg.model == "NaiveAvg":
        model = NaiveAverage()
    else:
        logging.error(f"Model {cfg.model} unknown.")
        return None
    return model


def nan(shape):
    ar = np.empty(shape)
    ar[:] = np.nan
    return ar


def get_avg_ts(ts, n_avg):
    ts_ret = nan(len(ts))
    for k in range(n_avg - 1, len(ts)):
        ts_ret[k] = np.sum(ts[k - n_avg + 1:k + 1]) / n_avg
    return ts_ret


class RVPredModel(Model):
    def __init__(self):
        super(RVPredModel, self).__init__()

    @property
    def channels_out(self):
        return 1

    @property
    def channels_in(self):
        return 1

    @property
    def is_tf_model(self):
        return True

    def vola_from_pred(self, inp_ts, pred):
        return pred

    def get_inp_ts(self, ts):
        return tf.reshape(tf.convert_to_tensor(ts, dtype=DTYPE), [1, len(ts[:, 0]), -1])

    def labels_from_inp_ts(self, inp_ts, idx_range):
        return inp_ts[:, idx_range[0]:idx_range[1], 0:1]

    def random_batch_generator(self, ts, idx_range_train, label_length, batch_size, steps_per_epoch, valid_idxs,
                               weights,
                               dtype=DTYPE):
        def generator(training=False):
            for k in range(steps_per_epoch):
                x = []
                y = []
                w = []
                for k2 in range(batch_size):
                    idx = -1
                    while idx < idx_range_train[0] or idx > idx_range_train[1] - label_length:
                        idx = valid_idxs[np.random.randint(0, len(valid_idxs))]
                    x.append(ts[:, idx - self.max_lag:idx + label_length - 1, :])
                    y.append(self.labels_from_inp_ts(ts, [idx, idx + label_length]))
                    w.append(weights[:, idx - idx_range_train[0]:idx + label_length - idx_range_train[0], 0:1])
                x = tf.concat(x, axis=0)
                y = tf.concat(y, axis=0)
                w = tf.concat(w, axis=0)
                yield x, y, w

        return generator


class NaiveAverage(object):

    def __init__(self, n_avg):
        self.n_avg = n_avg

    @property
    def max_lag(self):
        return self.n_avg

    def predict(self, ts):
        return get_avg_ts(ts, self.n_avg)[self.max_lag - 1:]

    def fit(self, ts):
        pass

    def save_weights(self, path):
        pass


class HAR(object):
    def __init__(self, lags, fit_method='OLS', clip_value=0.0):
        self.lags = lags
        self.fit_method = fit_method
        self.clip_value = clip_value

    def get_X(self, ts):
        X = np.zeros([len(ts) - self.max_lag + 1, len(self.lags) + 1])
        ts_avgs = np.array([get_avg_ts(ts, lag) for lag in self.lags])

        for k in range(np.shape(X)[0]):
            X[k, 0] = 1
            X[k, 1:] = ts_avgs[:, k + self.max_lag - 1]
        return X

    def get_linear_system_fit(self, ts):
        y = ts[self.max_lag:, 0]
        X = self.get_X(ts[:-1, 0])
        return X, y

    def fit(self, ts):
        X, y = self.get_linear_system_fit(ts)
        self.lm = LinearRegression(fit_intercept=False).fit(X, y)
        if self.fit_method == 'WLS':
            weights = 1 / self.lm.predict(X)
        elif self.fit_method == 'OLS':
            weights = 1.0
        else:
            logging.warning(f"Baseline fit {self.baseline_fit} unknown. Using weights = 1.0.")
            weights = 1.0
        self.lm = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=weights)

    def predict(self, ts):
        X = self.get_X(ts)
        return np.clip(self.lm.predict(X), a_min=self.clip_value, a_max=None)

    @property
    def max_lag(self):
        return np.max(self.lags)

    @property
    def coeffs(self):
        return self.lm.coef_

    @property
    def is_tf_model(self):
        return False

    def save(self, path):
        with open(path + "/lm.joblib", "w"):
            joblib.dump(self.lm, path + "/lm.joblib")
        np.save(path + "/clipping_value.npy", self.clip_value)

    def restore(self, path):
        with open(path + "/lm.joblib", "r"):
            self.lm = joblib.load(path + "/lm.joblib")
        self.clip_value = np.load(path + "/clipping_value.npy")


# 3 input channels
class HARSVJ(HAR):

    def __init__(self, lags, fit_method='OLS', clip_value=0.0):
        super(HARSVJ, self).__init__(lags, fit_method, clip_value)

    def get_linear_system_fit(self, ts):
        y = ts[self.max_lag:, 0]
        X = self.get_X(ts[:-1, :])
        return X, y

    def get_X(self, ts):
        # X = np.zeros([len(ts[:, 0]) - self.max_lag + 1, np.shape(ts)[1] + len(self.lags)] + 1)
        ts_avgs = []
        for k in range(np.shape(ts)[1] - 1):
            ts_avgs.append(np.array([get_avg_ts(ts[:, k], lag) for lag in self.lags]))
        X = np.transpose(np.concatenate(ts_avgs, 0))
        X = np.column_stack((X, ts[:, -1]))
        X = np.c_[np.ones(X.shape[0]), X]
        return X[self.max_lag - 1:, :]


class HARNet(RVPredModel):
    def __init__(self, filters_dconv, use_bias_dconv, activation_dconv, lags, regr_coeff, clip_value):
        super(HARNet, self).__init__()
        self.lags = lags
        self.clip_value = tf.Variable(float(clip_value), name="clip_value", trainable=False)
        self.har = HAR(lags)
        if np.any(np.array(self.lags)[1:] % np.array(self.lags)[:-1]):
            raise Exception('each lag must be a multiple of the previous one')
        self.coeffs = regr_coeff
        self.avg_layers = [Conv1D(filters=filters_dconv, kernel_size=self.lags[1],
                                  kernel_initializer=tf.keras.initializers.constant(1 / self.lags[1]),
                                  activation=activation_dconv, use_bias=use_bias_dconv, padding='causal')]
        for k in range(1, len(self.lags) - 1):
            self.avg_layers.append(
                Conv1D(filters=filters_dconv, kernel_size=int(self.lags[k + 1] / self.lags[k]),
                       padding='causal',
                       kernel_initializer=tf.keras.initializers.constant(self.lags[k] / self.lags[k + 1]),
                       dilation_rate=self.lags[k],
                       activation=activation_dconv, use_bias=False))
        self.output_layer = Dense(1, activation=None, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.constant(regr_coeff))

    @property
    def max_lag(self):
        return np.max(self.lags)

    def call(self, inputs, training=False):
        avgs = tf.concat([tf.ones_like(inputs), inputs], axis=-1)
        for avg_layer in self.avg_layers:
            avgs = tf.concat([avgs, avg_layer(inputs)], axis=-1)
            inputs = avgs[:, :, -1:]
        return tf.clip_by_value(self.output_layer(avgs)[:, self.max_lag - 1:, :],
                                clip_value_min=self.clip_value,
                                clip_value_max=1000)


def get_loss(name):
    if name == 'QLIKE':
        def qlike(y_true, y_pred):
            return tf.math.divide(y_true, y_pred) - tf.math.log(tf.math.divide(y_true, y_pred)) - 1

        return qlike

    else:
        return tf.keras.losses.get(name)


class HARNetSVJ(RVPredModel):
    def __init__(self, filters_dconv, use_bias_dconv, activation_dconv, lags, regr_coeff, clip_value):
        super(HARNetSVJ, self).__init__()
        self.lags = lags
        self.clip_value = tf.Variable(float(clip_value), name="clip_value", trainable=False)
        if np.any(np.array(self.lags)[1:] % np.array(self.lags)[:-1]):
            raise Exception('each lag must be a multiple of the previous one')
        self.coeffs = regr_coeff
        self.har = HARSVJ(lags)

        def kernel_avg_init(shape, dtype=DTYPE):
            kernel = np.zeros(shape)
            l = shape[0]
            kernel[:, :, 0] = np.transpose(np.concatenate([[np.repeat(1 / l, l)], [np.zeros(l)]]))
            kernel[:, :, 1] = np.transpose(np.concatenate([[np.zeros(l)], [np.repeat(1 / l, l)]]))
            return kernel

        self.avg_layers = [Conv1D(filters=filters_dconv, kernel_size=self.lags[1],
                                  kernel_initializer=kernel_avg_init,
                                  activation=activation_dconv, use_bias=use_bias_dconv, padding='causal')]

        for k in range(1, len(self.lags) - 1):
            self.avg_layers.append(
                Conv1D(filters=filters_dconv, kernel_size=int(self.lags[k + 1] / self.lags[k]),
                       padding='causal',
                       kernel_initializer=kernel_avg_init,
                       dilation_rate=self.lags[k],
                       activation=activation_dconv, use_bias=use_bias_dconv))
        regr_coeff_mapping = np.array([0])
        rv_avgs_idxs = np.arange(1, len(self.lags) + 1)
        rsv_avgs_idxs = np.arange(len(self.lags) + 1, len(self.lags) * 2 + 1)
        regr_coeff_mapping = np.append(regr_coeff_mapping, np.c_[rv_avgs_idxs, rsv_avgs_idxs].flatten())
        regr_coeff_mapping = np.append(regr_coeff_mapping, len(regr_coeff) - 1)  # to include regr coeff for jumps
        self.output_layer = Dense(1, activation='linear',
                                  kernel_initializer=tf.keras.initializers.constant(
                                      np.array([regr_coeff[k] for k in regr_coeff_mapping])),
                                  use_bias=False)

    @property
    def max_lag(self):
        return np.max(self.lags)

    @property
    def channels_in(self):
        return 3

    def call(self, inputs, training=False):
        jumps = tf.expand_dims(inputs[:, :, -1], -1)
        inputs = inputs[:, :, :-1]
        avgs = tf.concat([tf.ones([inputs.shape[0], inputs.shape[1], 1]), inputs], axis=-1)
        for avg_layer in self.avg_layers:
            avgs = tf.concat([avgs, avg_layer(inputs)], axis=-1)
            inputs = avgs[:, :, -2:]
        return tf.clip_by_value(self.output_layer(tf.concat([avgs, jumps], -1))[:, self.max_lag - 1:, :],
                                clip_value_min=self.clip_value,
                                clip_value_max=1000)


def get_pred(model, scaler, ts, pred_range=None):
    if pred_range is None:
        pred_range = [model.max_lag, len(ts[:, 0])]

    ts_norm = scaler.fit_transform(ts)
    target = ts[pred_range[0]:pred_range[1], 0]
    target_norm = ts_norm[pred_range[0]:pred_range[1], 0]
    if model.is_tf_model:
        ts_norm_in = model.get_inp_ts(ts_norm)
        ts_norm_pred_raw = model(ts_norm_in[:, pred_range[0] - model.max_lag:pred_range[1] - 1, :])
        ts_norm_pred = model.vola_from_pred(ts_norm_in[:, pred_range[0] - model.max_lag:pred_range[1] - 1, :],
                                            ts_norm_pred_raw).numpy().flatten()
        ts_norm_pred_raw = ts_norm_pred_raw.numpy()
        target_norm_raw = model.labels_from_inp_ts(ts_norm_in, pred_range).numpy()
    else:
        ts_norm_in = ts_norm[pred_range[0] - model.max_lag:pred_range[1] - 1, :]
        ts_norm_pred = model.predict(ts_norm_in)
        ts_norm_pred_raw = ts_norm_pred
        target_norm_raw = target_norm
    ts_pred = scaler.inverse_transform(ts_norm_pred)
    return ts_pred, ts_norm_pred, ts_norm_pred_raw, target, target_norm, target_norm_raw, pred_range


def get_model_metrics(model, scaler, ts, idx_range, loss=None, prefix=""):
    ts = ts[idx_range[0] - model.max_lag:idx_range[1], :]
    ts_pred, ts_norm_pred, ts_norm_pred_raw, target, target_norm, target_norm_raw, pred_range = get_pred(model, scaler,
                                                                                                         ts)
    return_dict = {}
    return_dict[prefix + "_MAE"] = np.mean(np.abs(ts_pred - target))
    return_dict[prefix + "_MSE"] = np.mean(np.square(ts_pred - target))
    return_dict[prefix + "_QLIKE"] = np.mean(tf.math.divide(target, ts_pred) -
                                             np.log(tf.math.divide(target, ts_pred)) - 1)
    return_dict[prefix + "_norm_MAE"] = np.mean(np.abs(ts_norm_pred - target_norm))
    return_dict[prefix + "_norm_MSE"] = np.mean(np.square(ts_norm_pred - target_norm))
    return_dict[prefix + "_norm_QLIKE"] = np.mean(
        tf.math.divide(target_norm, ts_norm_pred) - np.log(tf.math.divide(target_norm, ts_norm_pred)) - 1)

    if loss is not None:
        return_dict[prefix + "loss"] = tf.reduce_mean(loss(tf.convert_to_tensor(target_norm_raw, dtype=DTYPE),
                                                           tf.convert_to_tensor(ts_norm_pred_raw,
                                                                                dtype=DTYPE))).numpy()
    return return_dict


class MinMaxScaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, ts):
        return np.divide(ts - self.min, self.max - self.min)

    def fit_transform(self, ts):
        return self.transform(ts)

    def inverse_transform(self, ts):
        return np.multiply(ts, self.max - self.min) + self.min


class LogMinMaxScaler:
    def __init__(self, min, max):
        self.mm_scaler = MinMaxScaler(min, max)

    def transform(self, ts):
        return self.mm_scaler.transform(np.log(ts))

    def fit_transform(self, ts):
        return self.transform(ts)

    def inverse_transform(self, ts):
        return np.exp(self.mm_scaler.inverse_transform(ts))


class NoneScaler:
    def __init__(self):
        pass

    def transform(self, ts):
        return ts

    def fit_transform(self, ts):
        return self.transform(ts)

    def inverse_transform(self, ts):
        return ts


class LogScaler:
    def __init__(self):
        pass

    def transform(self, ts):
        return np.log(ts)

    def fit_transform(self, ts):
        return self.transform(ts)

    def inverse_transform(self, ts):
        return np.exp(ts)


class MetricCallback(Callback):
    def __init__(self, ts, idx_range_train, idx_range_val, scaler, tb_path, save_best_weights=None):
        self.ts = ts
        self.idx_range_train = idx_range_train
        self.idx_range_val = idx_range_val
        self.scaler = scaler
        self.tb_path = tb_path
        self.writer = tf.summary.create_file_writer(tb_path)
        self.best_weights = None
        self.best_loss = np.Inf
        self.best_epoch = 0
        self.save_best_weights = save_best_weights

    def on_epoch_begin(self, epoch, logs=None):
        metric_dict_train = get_model_metrics(self.model, self.scaler, self.ts, self.idx_range_train,
                                              loss=self.model.loss, prefix="train_")
        metric_dict_val = get_model_metrics(self.model, self.scaler, self.ts, self.idx_range_val,
                                            loss=self.model.loss, prefix="test_")

        metric_dict = {**metric_dict_train, **metric_dict_val}

        if self.save_best_weights in metric_dict.keys():
            if metric_dict[self.save_best_weights] < self.best_loss:
                self.best_loss = metric_dict[self.save_best_weights]
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch

        with self.writer.as_default():
            for key in metric_dict:
                tf.summary.scalar(key, metric_dict[key], step=epoch)
            self.writer.flush()

        for k, v in metric_dict.items():
            self.model.history.history.setdefault(k, []).append(v)

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            logger.debug(
                f"Train end: Set weights to epoch {self.best_epoch} with {self.save_best_weights} = {self.best_loss}.")


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
