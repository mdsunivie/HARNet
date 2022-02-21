#!/usr/bin/env python3
"""
HARNet CLI
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pydantic.json import pydantic_encoder
from tensorflow.python.keras import backend as K

from harnet.model import scaler_from_cfg, model_from_cfg, get_loss, LRTensorBoard, MetricCallback, get_model_metrics
from harnet.util import HARNetCfg, get_MAN_data, year_range_to_idx_range

pd.options.display.float_format = '{:,e}'.format
pd.options.display.width = 0


def main():
    # parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", default="config.in", nargs='?',
                        help="Base configuration file to use (e.g. config.in). Must be a JSON-file of a config-dict. The used configuration is a combination of defaults set in util.py and parameters set in this file.")
    args = parser.parse_args()

    # load configuration
    cfg = HARNetCfg()
    if os.path.exists(args.cfg):
        with open(args.cfg) as f:
            cfg_in = json.load(f)
        for key in cfg_in:
            setattr(cfg, key, cfg_in[key])

    exp_name = os.path.splitext(os.path.basename(args.cfg))[0]
    logger = logging.getLogger('harnet')

    logger.info(f"Initializing experiment {exp_name}: {cfg.epochs} ...")
    logger.debug(json.dumps(cfg, indent=4, default=pydantic_encoder))

    save_path_curr = os.path.join(cfg.save_path,exp_name)
    tb_path_curr = os.path.join(cfg.tb_path,exp_name)

    if not os.path.exists(Path(cfg.save_path)):
        os.makedirs(Path(cfg.save_path))


    if not os.path.exists(Path(save_path_curr)):
        os.makedirs(Path(save_path_curr))

    # save full config
    with open(os.path.join(save_path_curr, "cfg_full.in"), 'w') as f:
        json.dump(cfg, f, indent=4, default=pydantic_encoder)

    # copy original config file
    shutil.copyfile(args.cfg, os.path.join(save_path_curr, os.path.basename(args.cfg)))

    # load data
    ts = get_MAN_data(cfg.path_MAN, cfg.asset, cfg.include_sv)

    # normalize input time series
    scaler = scaler_from_cfg(cfg)
    ts_norm = pd.DataFrame(data=scaler.fit_transform(ts.to_numpy()), index=ts.index)

    # create train datasets
    year_range_train = [cfg.start_year_train, cfg.start_year_train + cfg.n_years_train]
    year_range_test = [cfg.start_year_train + cfg.n_years_train,
                      cfg.start_year_train + cfg.n_years_train + cfg.n_years_test]

    idx_range_train = year_range_to_idx_range(ts_norm, year_range_train)
    idx_range_test = year_range_to_idx_range(ts_norm, year_range_test)

    # init model
    model = model_from_cfg(cfg, ts_norm, scaler, idx_range_train)

    # fit model
    if not model.is_tf_model:
        print(f"\n-- Fitting {exp_name}... --")
        model.fit(ts_norm.values[idx_range_train[0] - model.max_lag:idx_range_train[1], :])
        model.save(save_path_curr)
    else:
        print(f"\n-- Fitting {exp_name} with {cfg.epochs} epochs... --")
        optimizer = tf.keras.optimizers.get(cfg.optimizer)
        K.set_value(optimizer.lr, cfg.learning_rate)

        # hier richtige inp ts and prediction uebergeben
        ts_norm_in = model.get_inp_ts(ts_norm.values)

        model.compile(optimizer=optimizer, loss=get_loss(cfg.loss), sample_weight_mode="temporal")

        callbacks = []
        callbacks.append(LRTensorBoard(log_dir=tb_path_curr, profile_batch=0))
        callbacks.append(MetricCallback(ts.to_numpy(), idx_range_train, idx_range_test, scaler, tb_path_curr,
                                        save_best_weights=cfg.save_best_weights))
        model.run_eagerly = cfg.run_eagerly

        if cfg.baseline_fit == 'WLS':
            weights = 1 / model(ts_norm_in[:, idx_range_train[0] - model.max_lag:idx_range_train[1] - 1, :])
        else:
            weights = tf.ones_like(ts_norm_in[:, idx_range_train[0]:idx_range_train[1], :])

        valid_batch_gen_idxs = list(
            range(idx_range_train[0] + model.max_lag, idx_range_train[1] - cfg.label_length + 1))
        ds_fit = tf.data.Dataset.from_generator(
            model.random_batch_generator(ts_norm_in[:, :idx_range_train[1], :], idx_range_train,
                                         cfg.label_length,
                                         cfg.batch_size, cfg.steps_per_epoch,
                                         valid_batch_gen_idxs, weights),
            (tf.float32, tf.float32, tf.float32), output_shapes=(
                tf.TensorShape(
                    [cfg.batch_size, model.max_lag + cfg.label_length - 1,
                     model.channels_in]),
                tf.TensorShape(
                    [cfg.batch_size, cfg.label_length, model.channels_out]),
                tf.TensorShape(
                    [cfg.batch_size, cfg.label_length, model.channels_out])
            ))

        history = model.fit(ds_fit, epochs=cfg.epochs, verbose=cfg.verbose,
                            callbacks=callbacks)  # [TqdmCallback(verbose=1)]

        # plot optimization
        # dpv.plot_values(list(history.history.values()), None, fmt="-", logx=False,
        #                 title_str="Optimization History for Model %s" % config.model.name,
        #                 labels=list(history.history.keys()))
        # plt.gcf().savefig(Path(save_path_curr + "/optimization.pdf"))
        # model.summary()
        chkpt_model = tf.train.Checkpoint(model=model)
        chkpt_model.write(save_path_curr + "/model_params")
        pd.DataFrame.from_dict(history.history).to_csv(save_path_curr + '/metrics_history.csv', index=False)

    metrics_train = get_model_metrics(model, scaler, ts.to_numpy(), idx_range_train, prefix='train')  #
    df_metrics_train = pd.DataFrame(metrics_train, index=[cfg.model])
    metrics_test = get_model_metrics(model, scaler, ts.to_numpy(), idx_range_test, prefix='test')  #
    df_metrics_test = pd.DataFrame(metrics_test, index=[cfg.model])
    df_metrics = pd.concat([df_metrics_train, df_metrics_test], axis=1)
    print("")
    print(df_metrics)
    df_metrics.to_csv(save_path_curr + "/metrics.csv")
    print(f"\n-- Experiment finished. Results were saved to {save_path_curr} --\n")
