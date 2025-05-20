# 🧩 API Definition for Forecasting

### 1. 🤖 Model API

- Python file name

    Adapter model class should be placed in a file under `models/YOUR_MODEL_NAME.py`, where `YOUR_MODEL_NAME` is to be provided in the `--model_name` argument helping the pipeline import the model class automatically.

    Other model dependencies are encouraged to be put under `models/layers/YOUR_MODEL_NAME/` folder
- Adapter class name

    The outer model class should be renamed as `Model`.
    ```python
    class Model(nn.Module):
    ```

- `__init__()`

    Minimal example:

    ```python
    def __init__(
        self,
        configs: ExpArgs
    ):
        super().__init__()
    ```

    Existing arguments can be found in `utils/args.py`, and `utils/ExpArgs.py` is used to support pylint checking.

    > ❗️The global configuration should be treated as **read only**.

- `forward()` input arguments

    Minimal example:

    ```python
    def forward(
        self, 
        x: Tensor, # mandatory; lookback sequence; (batch_size, seq_len, enc_in)
        y: Tensor = None, # mandatory; forecast groundtruth; (batch_size, pred_len, c_out)
        **kwargs # mandatory; container for redundant input parameters
    ):
    ```

    Other available args:

    ```python
    x_mark: Tensor = None, # lookback timestamps; (batch_size, seq_len, enc_in)
    x_mask: Tensor = None, # lookback mask; (batch_size, seq_len, enc_in)
    y_mark: Tensor = None, # forecast timestamps; (batch_size, pred_len, c_out)
    y_mask: Tensor = None, # forecast mask; (batch_size, pred_len, c_out)
    exp_stage: str = "train", # indicator for train/val/test
    ```

- `forward()` return value

    Minimal example: 

    ```python
    return {
        "pred": ..., # model's output, should be of same shape as "true"
        "true": ..., # groundtruth. Normally it is "y"
    }
    ```

    Other available items:

    ```python
    "mask": ..., # mask for groundtruth. Normally it is "y_mask"
    "loss": ..., # commonly used with "ModelProvidedLoss"
    ```

### 2. 💾 Dataset API

- Python file name

    Adapter dataset class should be placed in a file under `data/data_provider/datasets/YOUR_DATASET_NAME.py`, where `YOUR_DATASET_NAME` is to be provided in the `--dataset_name` argument helping the pipeline import the dataset class automatically.

    Other model dependencies are encouraged to be put under `data/dependencies/YOUR_DATASET_NAME/` folder
- Adapter class name

    The outer dataset class should be renamed as `Data`.
    ```python
    class Data(Dataset):
    ```

- `__init__()`

    Minimal example:

    ```python
    def __init__(
        self, 
        configs: ExpArgs,
        flag: str = 'train'
    ):
    ```

    Existing arguments can be found in `utils/args.py`, and `utils/ExpArgs.py` is used to support pylint checking.

    > ❗️The global configuration should be treated as **read only**.

- `__getitem__()` return value

    Minimal example:
    
    ```python
    return {
        "x": ..., # mandatory; lookback sequence; (seq_len, enc_in)
        "y": ..., # mandatory; forecast groundtruth; (pred_len, c_out)
    }
    ```

    Other available items:

    ```python
    "x_mark": ..., # lookback timestamps; (seq_len, enc_in)
    "x_mask": ..., # lookback mask; (seq_len, enc_in)
    "y_mark": ..., # forecast timestamps; (pred_len, c_out)
    "y_mask": ..., # forecast mask; (pred_len, c_out)
    "sample_ID": ..., # sample ID; (1)
    ```

### 3. 📉 Loss Function API

- Python file name

    Loss function class should be placed in a file under `loss_fns/YOUR_LOSS_FN.py`, where `YOUR_LOSS_FN` is to be provided in the `--loss` argument helping the pipeline import the loss function class automatically.

- Adapter class name

    The loss function class should be renamed as `Loss`.
    ```python
    class Loss(nn.Module):
    ```

- `forward()` input arguments

    Minimal example:

    ```python
    def forward(
        self, 
        **kwargs # mandatory; container for redundant input parameters
    ):
    ```

    Other available args:

    ```python
    pred, # model's output, should be of same shape as "true"
    true, # groundtruth. Normally it is "y"
    mask, # mask for groundtruth. Normally it is "y_mask"
    loss, # commonly used with "ModelProvidedLoss"
    ```

- `forward()` return value

    Minimal example: 

    ```python
    return {
        "loss": ..., # of shape (1)
    }
    ```

