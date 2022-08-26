import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def get_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def get_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_r(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num = y_true.shape[0]
    sum_1 = np.sum([(y_true[i] - np.mean(y_true)) * (y_pred[i] - np.mean(y_pred)) for i in range(num)])
    sum_2 = np.sum([np.square(y_true[i] - np.mean(y_true)) for i in range(num)])
    sum_3 = np.sum([np.square(y_pred[i] - np.mean(y_pred)) for i in range(num)])
    r = (sum_1 + 1e-32) / (np.sqrt(sum_2 * sum_3) + 1e-32)
    return r

# print(get_r([1, 2, 3], [1, 2, 3.01]))

def get_ia(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num = y_true.shape[0]
    sum_1 = np.sum([np.square(y_true[i] - y_pred[i]) for i in range(num)])
    sum_2 = np.sum([np.square(np.abs(y_true[i] - np.mean(y_true)) + np.abs(y_pred[i] - np.mean(y_pred))) for i in range(num)])
    ia = 1 - ((sum_1 + 1e-32) / (sum_2 + 1e-32))
    return ia

# print(get_ia([1, 2, 3], [1, 2, 2]))

def get_u95(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num = y_true.shape[0]
    ee = y_true - y_pred
    u95 = 1.96 * np.sqrt(np.var(ee) + mean_squared_error(y_true, y_pred))
    return u95

# print(get_u95([1, 2, 3], [1, 2, 3.01]))

def get_mre(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num = y_true.shape[0]
    mre_mean = np.sum([((np.abs(y_true[i] - y_pred[i]) + 1e-32) / (y_true[i] + 1e-32)) for i in range(num)]) / num
    # print(mre)
    return mre_mean

# print(get_mre([1, 2, 3], [1, 2, 3.01]))