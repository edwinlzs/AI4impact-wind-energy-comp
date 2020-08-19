import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Flatten,Input
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.optimizers import Adam

%matplotlib inline
