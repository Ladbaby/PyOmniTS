<div align="center">
  <img src="images/icon_dark.png#gh-dark-mode-only" height=200>
  <img src="images/icon_light.png#gh-light-mode-only" height=200>
  <h3><b> A Researcher-Friendly Framework for Time Series Analysis. </b></h3>
  <h4><b> Train Any Model on Any Dataset. </b></h4>
</div>

---

This is also the official repository for the following paper:

- HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting (ICML 2025) [[poster]](https://icml.cc/virtual/2025/poster/43741) [[OpenReview]](https://openreview.net/forum?id=u8wRbX2r2V) [[arXiv]](https://arxiv.org/abs/2505.17431)

## 1. ✨ Hightlighted Features

![](images/overview.png)

- **Extensibility**: Adapt your model/dataset **once**, train almost **any combination** of "model" $\times$ "dataset" $\times$ "loss function".
- **Compatibility**: Accept models with any number/type of arguments in `forward`; Accept datasets with any number/type of return values in `__getitem__`; Accept tailored loss calculation for specific models.
- **Maintainability**: No need to worry about breaking the training codes of existing models/datasets/loss functions when adding new ones.
- **Reproducibility**: Minimal library dependencies for core components. Try the best to get rid of fancy third-party libraries (e.g., Pytorch Lightning, EasyTorch).
- **Efficiency**: Multi-GPU parallel training; Python built-in logger; structured experimental result saving (json)...
- **Transferability**: Even if you don't like our framework, you can still easily find and copy the models/datasets you want. No overwhelming encapsulation.

## 2. 🧭 Documentation

1. [🚀 Get Started](https://github.com/Ladbaby/PyOmniTS/blob/master/docs/tutorial/1_get_started.md)
2. [⚙️ Change Experiment Settings](https://github.com/Ladbaby/PyOmniTS/blob/master/docs/tutorial/2_change_experiment_settings.md)
3. 🧩 API Definition

    - [Forecasting API](https://github.com/Ladbaby/PyOmniTS/blob/master/docs/forecasting/1_API.md)

## 3. 🤖 Models

44 models, covering regular, irregular, pretrained, and traffic models, have been included in PyOmniTS, and more are coming.

Model classes can be found in `models/`, and their dependencies can be found in `layers/`

- ✅: supported
- ❌: not supported
- '-': not implemented
- MTS: regularly sampled multivariate time series
- IMTS: irregularly sampled multivariate time series

|Model|Venue|Type|Forecasting|Classification|Imputation
|---|---|---|---|---|---|
|[Ada-MSHyper](https://openreview.net/forum?id=RNbrIQ0se8)|NeurIPS 2024|MTS|✅|-|-
|[Autoformer](https://openreview.net/pdf?id=I55UqU-M11y)|NeurIPS 2021|MTS|✅|✅|✅
|[BigST](https://dl.acm.org/doi/abs/10.14778/3641204.3641217)|VLDB 2024|MTS|✅|-|-
|[Crossformer](https://openreview.net/pdf?id=vSVLM2j9eie)|ICLR 2023|MTS|✅|✅|✅
|[CRU](https://proceedings.mlr.press/v162/schirmer22a.html)|ICML 2022|IMTS|✅|❌|-
|[DLinear](https://ojs.aaai.org/index.php/AAAI/article/view/26317)|AAAI 2023|MTS|✅|✅|-
|[ETSformer](https://arxiv.org/abs/2202.01381)|arXiv 2022|MTS|✅|✅|-
|[FEDformer](https://proceedings.mlr.press/v162/zhou22g.html)|ICML 2022|MTS|✅|✅|-
|[FiLM](https://papers.nips.cc/paper_files/paper/2022/hash/524ef58c2bd075775861234266e5e020-Abstract-Conference.html)|NeurIPS 2022|MTS|✅|✅|-
|[FourierGNN](https://openreview.net/forum?id=bGs1qWQ1Fx)|NeurIPS 2023|MTS|✅|-|-
|[FreTS](https://papers.nips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html)|NeurIPS 2023|MTS|✅|✅|-
|[GNeuralFlows](https://openreview.net/forum?id=tFB5SsabVb)|NeurIPS 2024|IMTS|✅|❌|✅
|[GraFITi](https://ojs.aaai.org/index.php/AAAI/article/view/29560)|AAAI 2024|IMTS|✅|-|✅
|[GRU-D](https://www.nature.com/articles/s41598-018-24271-9)|Scientific Reports 2018|IMTS|✅|✅|✅
|[Hi-Patch](https://openreview.net/forum?id=nBgQ66iEUu)|ICML 2025|IMTS|✅|✅|-
|[higp](https://proceedings.mlr.press/v235/cini24a.html)|ICML 2024|MTS|✅|-|-
|[HyperIMTS](https://openreview.net/forum?id=u8wRbX2r2V)|ICML 2025|IMTS|✅|-|-
|[Informer](https://ojs.aaai.org/index.php/AAAI/article/view/17325)|AAAI 2021|MTS|✅|✅|✅
|[iTransformer](https://openreview.net/forum?id=JePfAI8fah)|ICLR 2024|MTS|✅|✅|-
|[Koopa](https://papers.nips.cc/paper_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html)|NeurIPS 2023|MTS|✅|❌|-
|[Latent_ODE](https://papers.nips.cc/paper_files/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html)|NeurIPS 2019|IMTS|✅|❌|-
|[Leddam](https://openreview.net/forum?id=87CYNyCGOo)|ICML 2024|MTS|✅|✅|-
|[LightTS](https://arxiv.org/abs/2207.01186)|arXiv 2022|MTS|✅|✅|-
|[Mamba](https://openreview.net/forum?id=tEYskw1VY2)|Language Modeling 2024|MTS|✅|✅|-
|[MICN](https://openreview.net/pdf?id=zt53IDUR1U)|ICLR 2023|MTS|✅|✅|-
|[MOIRAI](https://proceedings.mlr.press/v235/woo24a.html)|ICML 2024|Any|✅|-|-
|[mTAN](https://openreview.net/forum?id=4c0J6lwQ4_)|ICLR 2021|IMTS|✅|✅|✅
|[NeuralFlows](https://proceedings.neurips.cc/paper/2021/hash/b21f9f98829dea9a48fd8aaddc1f159d-Abstract.html)|NeurIPS 2021|IMTS|✅|❌|-
|[Nonstationary Transformer](https://openreview.net/pdf?id=ucNDIDRNjjv)|NeurIPS 2022|MTS|✅|✅|-
|[PatchTST](https://openreview.net/forum?id=Jbdc0vTOcol)|ICLR 2023|MTS|✅|✅|✅
|[PrimeNet](https://ojs.aaai.org/index.php/AAAI/article/view/25876)|AAAI 2023|IMTS|✅|✅|-
|[Pyraformer](https://openreview.net/pdf?id=0EXmFzUn5I)|ICLR 2022|MTS|✅|✅|-
|[Raindrop](https://openreview.net/forum?id=Kwm8I7dU-l5)|ICLR 2022|IMTS|✅|✅|✅
|[Reformer](https://openreview.net/forum?id=rkgNKkHtvB)|ICLR 2020|MTS|✅|✅|-
|[SeFT](https://proceedings.mlr.press/v119/horn20a.html)|ICML 2020|IMTS|✅|✅|✅
|[SegRNN](https://arxiv.org/abs/2308.11200)|arXiv 2023|MTS|✅|✅|-
|[Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363)|arXiv 2019|MTS|✅|-|-
|[TiDE](https://openreview.net/forum?id=pCbC3aQB5W)|TMLR 2023|MTS|✅|✅|-
|[TimeMixer](https://openreview.net/forum?id=7oLshfEIC2)|ICLR 2024|MTS|✅|✅|-
|[TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq)|ICLR 2023|MTS|✅|✅|-
|[tPatchGNN](https://openreview.net/forum?id=UZlMXUGI6e)|ICML 2024|IMTS|✅|-|✅
|[Transformer](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)|NeurIPS 2017|MTS|✅|✅|-
|[TSMixer](https://openreview.net/forum?id=wbpxTuXgm0)|TMLR 2023|MTS|✅|✅|-
|[Warpformer](https://dl.acm.org/doi/abs/10.1145/3580305.3599543)|KDD 2023|IMTS|✅|-|-


## 4. 💾 Datasets

Dataest classes are put in `data/data_provider/datasets`, and dependencies can be found in `data/dependencies`:

11 datasets, covering regular and irregular ones, have been included in PyOmniTS, and more are coming.

- ✅: supported
- ❌: not supported
- '-': not implemented
- MTS: regularly sampled multivariate time series
- IMTS: irregularly sampled multivariate time series

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

Datasets for classification and imputation have not released yet.

## 5. 📉 Loss Functions

The following loss functions are included under `loss_fns/`:

|Loss Function|Task|Note
|---|---|---|
|CrossEntropyLoss|Classification|-|
|MAE|Forecasting/Imputation|-|
|ModelProvidedLoss|-|Some models prefer to calculate loss within `forward()`, such as GNeuralFlows.|
|MSE_Dual|Forecasting/Imputation||Used in Ada-MSHyper|
|MSE|Forecasting/Imputation|-|

## 6. 🚧 Roadmap

PyOmniTS is continously evolving:

- [ ] More tutorials.
- [ ] Classification support in core components.
- [ ] Imputation support in core components.
- [ ] Optional python package management via [uv](https://github.com/astral-sh/uv).

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

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ladbaby"><img src="https://avatars.githubusercontent.com/u/82816258?v=4?s=100" width="100px;" alt="Ladbaby"/><br /><sub><b>Ladbaby</b></sub></a><br /><a href="#code-Ladbaby" title="Code">💻</a> <a href="#bug-Ladbaby" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Acknowledgement

- [Time Series Library](https://github.com/thuml/Time-Series-Library): Models and datasets for regularly sampled time series are mostly adapted from it.
- [BasicTS](https://github.com/GestaltCogTeam/BasicTS): Documentation design reference.
- [Google Gemini](https://gemini.google.com/): Icon creation.

