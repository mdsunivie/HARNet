# Description

TensorFlow implementation of the HARNet model for realized volatility forecasting.

# Publication

R. Reisenhofer, X. Bayer, and N. Hautsch <br /> 
HARNet: A Convolutional Neural Network for Realized Volatility Forecasting <br />
arXiv preprint arXiv:2205.07719, 2022 <br />
[https://doi.org/10.48550/arXiv.2205.07719](https://doi.org/10.48550/arXiv.2205.07719)

Please cite the paper above when using the HARNet package in your research.

# Installation

Clone the repository and use

```bash
pip install -e HARNet/
```

to install the package.

# Usage

Download the MAN file and save it to HARNet/data.

Go to the HARNet root directory
```bash
cd HARNet
```
an start single experiments based on one of the preset configuration files

```bash
harnet ./configs/RV/RecField_20/HAR20_OLS.in
harnet ./configs/RV/RecField_20/QLIKE/HARNet20_QLIKE_OLS.in
```

Start experiments for all preset configuration files

```bash
/bin/bash run_all.sh
```

Results and TensorBoards for all experiments are saved in the ./HARNet/results folder.

# About

The HARNet package was developed by Rafael Reisenhofer and Xandro Bayer.

If you have any questions, please contact [rafael.reisenhofer@uni-bremen.de](mailto:rafael.reisenhofer@uni-bremen.de).
