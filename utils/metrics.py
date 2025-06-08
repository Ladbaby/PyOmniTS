import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)

def MAE(pred, true, mask=None):
    if mask is None:
        return np.mean(np.abs(pred - true))
    else:
        residual = (pred - true) * mask
        num_eval = np.sum(mask)
        return np.sum(np.abs(residual)) / (num_eval if num_eval > 0 else 1)

def MSE(pred, true, mask=None):
    if mask is None:
        return np.mean((pred - true) ** 2)
    else:
        residual = (pred - true) * mask
        num_eval = np.sum(mask)
        return np.sum(residual ** 2) / (num_eval if num_eval > 0 else 1)

def RMSE(pred, true, mask=None):
    return np.sqrt(MSE(pred, true, mask))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def metric(
    pred: np.ndarray, 
    y: np.ndarray, 
    y_mask: np.ndarray = None,
    **kwargs
):
    mae = MAE(pred, y, y_mask)
    mse = MSE(pred, y, y_mask)

    mse_per_sample = []
    for sample in range(pred.shape[0]):
        if y_mask is not None:
            mse_per_sample.append(MSE(pred[sample], y[sample], y_mask[sample]))
        else:
            mse_per_sample.append(MSE(pred[sample], y[sample]))

    # rmse = RMSE(pred, true, mask)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    # rse = RSE(pred, true)
    # corr = CORR(pred, true)

    # return mae, mse, rmse, mape, mspe, rse, corr
    return {
        "MAE": mae,
        "MSE": mse,
        # "MSE_per_sample": mse_per_sample
    }
