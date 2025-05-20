<div align="center">
  <img src="images/icon_dark.png#gh-dark-mode-only" height=200>
  <img src="images/icon_light.png#gh-light-mode-only" height=200>
  <h3><b> A Researcher-Friendly Framework for Time Series Analysis. </b></h3>
  <h4><b> Train Any Model on Any Dataset. </b></h4>
</div>

---

This is also the official repository for the following paper:

- [HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](https://openreview.net/forum?id=u8wRbX2r2V) (ICML 2025)

## 1. ✨ Hightlighted Features

![](images/overview.png)

- **Extensibility**: Adapt your model/dataset **once**, train almost **any combination** of "model" $\times$ "dataset" $\times$ "loss function".
- **Compatibility**: Accept models with any number/type of arguments in `forward`; Accept datasets with any number/type of return values in `__getitem__`; Accept tailored loss calculation for specific models.
- **Maintainability**: No need to worry about breaking the training codes of existing models/datasets/loss functions when adding new ones.
- **Reproducibility**: Minimal library dependencies for core components. Try the best to get rid of fancy third-party libraries (e.g., Pytorch Lightning, EasyTorch).
- **Efficiency**: Multi-GPU parallel training; Python built-in logger; structured experimental result saving (json)...
- **Transferability**: Even if you don't like our framework, you can still easily find and copy the models/datasets you want. No overwhelming encapsulation.

## 2. 🧭 Documentation

1. [🚀 Get Started](https://github.com/qianlima-lab/PyOmniTS/blob/master/docs/tutorial/1_get_started.md)
2. 🧩 API Definition

    - [Forecasting API](https://github.com/qianlima-lab/PyOmniTS/blob/master/docs/forecasting/1_API.md)

## 3. 🤖 Models

44 models, covering regular, irregular, pretrained, and traffic models, have been included in PyOmniTS, and more are coming.

Model classes can be found in `models/`, and their dependencies can be found in `layers/`

- ✅: supported
- ❌: not supported
- '-': not implemented

|Model|Venue|Type|Forecasting|Classification|Imputation
|---|---|---|---|---|---|
|Ada-MSHyper|NeurIPS 2024|MTS|✅|-|-
|Autoformer|NeurIPS 2021|MTS|✅|✅|-
|BigST|VLDB 2024|MTS|✅|-|-
|Crossformer|ICLR 2023|MTS|✅|✅|-
|CRU|ICML 2022|IMTS|✅|❌|-
|DLinear|AAAI 2023|MTS|✅|✅|-
|ETSformer|arXiv 2022|MTS|✅|✅|-
|FEDformer|ICML 2022|MTS|✅|✅|-
|FiLM|NeurIPS 2022|MTS|✅|✅|-
|FourierGNN|NeurIPS 2023|MTS|✅|-|-
|FreTS|NeurIPS 2023|MTS|✅|✅|-
|GNeuralFlows|NeurIPS 2024|IMTS|✅|❌|-
|GraFITi|AAAI 2024|IMTS|✅|-|-
|GRU-D|Scientific Reports 2018|IMTS|✅|✅|-
|Hi-Patch|ICML 2025|IMTS|✅|✅|-
|higp|ICML 2024|MTS|✅|-|-
|HyperIMTS|ICML 2025|IMTS|✅|-|-
|Informer|AAAI 2021|MTS|✅|✅|-
|iTransformer|ICLR 2024|MTS|✅|✅|-
|Koopa|NeurIPS 2023|MTS|✅|❌|-
|Latent_ODE|NeurIPS 2019|IMTS|✅|❌|-
|Leddam|ICML 2024|MTS|✅|✅|-
|LightTS|arXiv 2022|MTS|✅|✅|-
|Mamba|Language Modeling 2024|MTS|✅|✅|-
|MICN|ICLR 2023|MTS|✅|✅|-
|MOIRAI|ICML 2024|Any|✅|-|-
|mTAN|ICLR 2021|IMTS|✅|✅|-
|NeuralFlows|NeurIPS 2021|IMTS|✅|❌|-
|Nonstationary Transformer|NeurIPS 2022|MTS|✅|✅|-
|PatchTST|ICLR 2023|MTS|✅|✅|-
|PrimeNet|AAAI 2023|IMTS|✅|✅|-
|Pyraformer|ICLR 2022|MTS|✅|✅|-
|Raindrop|ICLR 2022|IMTS|✅|✅|-
|Reformer|ICLR 2020|MTS|✅|✅|-
|SeFT|ICML 2020|IMTS|✅|✅|-
|SegRNN|arXiv 2023|MTS|✅|✅|-
|Temporal Fusion Transformer|arXiv 2019|MTS|✅|-|-
|TiDE|TMLR 2023|MTS|✅|✅|-
|TimeMixer|ICLR 2024|MTS|✅|✅|-
|TimesNet|ICLR 2023|MTS|✅|✅|-
|tPatchGNN|ICML 2024|IMTS|✅|-|-
|Transformer|NeurIPS 2017|MTS|✅|✅|-
|TSMixer|TMLR 2023|MTS|✅|✅|-
|Warpformer|KDD 2023|IMTS|✅|-|-


## 4. 💾 Datasets

Dataest classes are put in `data/data_provider/datasets`, and dependencies can be found in `data/dependencies`:

11 datasets, covering regular and irregular ones, have been included in PyOmniTS, and more are coming.

- ✅: supported
- ❌: not supported
- '-': not implemented

|Dataset|Type|Field|Forecasting
|---|---|---|---|
|ECL|MTS|electricity|✅
|ETTh1|MTS|electricity|✅
|ETTm1|MTS|electricity|✅
|Human Activity|IMTS|biomechanics|✅
|ILI|MTS|healthcare|✅
|MIMIC III|IMTS|healthcare|✅
|MIMIC IV|IMTS|healthcare|✅
|PhysioNet'12|IMTS|healthcare|✅
|Traffic|MTS|traffic|✅
|USHCN|IMTS|weather|✅
|Weather|MTS|weather|✅

## 5. 📉 Loss Functions

The following loss functions are included under `loss_fns/`:

|Loss Function|Task|Note
|---|---|---|
|CrossEntropyLoss|Classification|-|
|MAE|Forecasting/Imputation|-|
|ModelProvidedLoss|-|Some models prefer to calculate loss within `forward()`, such as GNeuralFlows.|
|MSE_Dual|Forecasting/Imputation||Used in Ada-MSHyper|
|MSE|Forecasting/Imputation|-|

## 6. Roadmap

PyOmniTS is continous evolving:

- [ ] More tutorials.
- [ ] Classification support in core components.
- [ ] Optional python package management via uv.

## Yet Another Code Framework?

We encountered the following problems when using existing ones:

- Argument & return value chaos for **models**' `forward()`: 

    Different models usually take varying number and shape of arguments, especially ones from different domains. 
    Changes to training logic are needed to support these differences.
- Return value chaos for **datasets**' `__getitem__()`: 

    datasets can return a number of tensors in different shapes, which have to be aligned with arguments of models' `forward()` one by one.
    Changes to training logic are also needed to support these differences.
- Argument & return value chaos for **loss functions**' `forward()`: 

    loss functions take different types of tensors as input, require aligning with return values from models' `forward()`.
- Overwhelming dependencies: 

    some existing pipelines use fancy high-level packages in building the pipeline, which can lower the flexibility of code modification.

## Acknowledgement

- [Time Series Library](https://github.com/thuml/Time-Series-Library): Models and datasets for regularly sampled time series are mostly adapted from it.
- [BasicTS](https://github.com/GestaltCogTeam/BasicTS): Documentation design reference.
- [Google Gemini](https://gemini.google.com/): Icon creation.

