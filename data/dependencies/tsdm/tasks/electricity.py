r"""Tasks associated with the Electricity dataset."""


__all__ = [
    # CLASSES
    "ElectricityDeepState",
    "ElectricityDeepAR",
    "ElectricityTRMF",
    "ElectricityELBMBTTF",
]


class ElectricityDeepState:
    r"""Experiments as performed by the "DeepState" paper.

    Paper
    -----
    - | Deep State Space Models for Time Series Forecasting
      | Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
        Tim Januschowski
      | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
      | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html

    Evaluation Protocol
    -------------------

    .. epigraph::

        We train each method on all time series of these dataset but vary the size of the training
        range $Tᵢ∈\{14, 21, 28\}$ days. We evaluate all the methods on the next $τ=7$ days after the
        forecast start time using the standard p50 and p90- quantile losses.

    Test-Metric
    -----------

    Results
    -------
    Observation horizons: [14, 21, 28] days
    Forecast    horizons: 7 days
    Split:

    NBEATS claims a key at 2014-09-01 is used. But this seems wrong.
    The date 2014-09-01 only ever occurs in Appendix A5, Figure 4&5 which show an example plot.
    """


class ElectricityDeepAR:
    r"""Experiments as performed by the "DeepAR" paper.

    Paper
    -----
    - | `DeepAR: Probabilistic forecasting with autoregressive recurrent networks
        <https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_

    Evaluation Protocol
    -------------------
    .. epigraph::

        For electricity we train with data between 2014-01-01 and 2014-09-01, for traffic we train
        all the data available before 2008-06-15. The results for electricity and traffic are
        computed using rolling window predictions done after the last point seen in training as
        described in [23]. We do not retrain our model for each window, but use a single model
        trained on the data before the first prediction window.

    Test-Metric
    -----------

    Results
    -------
    """


class ElectricityTRMF:
    r"""Experiments as performed by the "TRMF" paper.

    Paper
    -----
    - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
      | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html

    Evaluation Protocol
    -------------------

    .. epigraph::

        5.1 Forecasting
        [...]
        For electricity and traffic, we consider the 24-hour ahead forecasting task and use last
        seven days as the test periods.

        A.1 Datasets and Evaluation Criteria
        [...]
        electricity 3 : the electricity usage in kW recorded every 15 minutes, for n = 370 clients.
        We convert the data to reflect hourly consumption, by aggregating blocks of 4 columns,
        to obtain T = 26, 304. Teh coefficient of variation for electricity is 6.0341.

    Test-Metric
    -----------
    **Normalized deviation (ND)**

    .. math::
        𝖭𝖣(Y, Ŷ) = \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t) ∈ Ω_\text{test}}|Ŷ_{it}-Y_{it}|\Big)
        \Big/ \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t) ∈ Ω_\text{test}}|Y_{it}|\Big)

    **Normalized RMSE (NRMSE)**

    .. math::
        𝖭𝖱𝖬𝖲𝖤(Y, Ŷ) = \sqrt{\frac{1}{|Ω_\text{test}|}∑_{(i,t) ∈ Ω_\text{test}}|Ŷ_{it}-Y_{it}|^2}
        \Big/ \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t) ∈ Ω_\text{test}}|Y_{it}|\Big)

    Results
    -------
    +-------+-------+-------------+-------------+---------------+
    | Model | TRMF  | N-BEATS (G) | N-BEATS (I) | N-BEATS (I+G) |
    +=======+=======+=============+=============+===============+
    | ND    | 0.255 | 0.171       | 0.185       | 0.111         |
    +-------+-------+-------------+-------------+---------------+
    """


class ElectricityELBMBTTF:
    r"""Experiments as performed by the "LogSparseTransformer" paper.

    Paper
    -----
    - | Enhancing the Locality and Breaking the Memory Bottleneck of Transformer
        on Time Series Forecasting
      | Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, Xifeng Yan
      | Advances in Neural Information Processing Systems 32 (NeurIPS 2019)
      | https://proceedings.neurips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html

    Evaluation Protocol
    -------------------

    .. epigraph::

        For short-term forecasting, we evaluate rolling-day forecasts for seven days ( i.e.,
        prediction horizon is one day and forecasts start time is shifted by one day after
        evaluating the prediction for the current day [6]). For long-term forecasting, we directly
        forecast 7 days ahead.

        A.2 Training
        [...]
        For electricity-c and traffic-c, we take 500K training windows while for electricity-f and
        traffic-f, we select 125K and 200K training windows, respectively.

        A.3 Evaluation
        Following the experimental settings in [6], one week data from 9/1/2014 00:00 (included) 9
        on electricity-c and 6/15/2008 17:00 (included) 10 on traffic-c is left as test sets.
        For electricity-f and traffic-f dataset, one week data from 8/31/2014 00:15 (included) and
        6/15/2008 17:00 (included) 11 is left as test sets, respectively.

    Test-Metric
    -----------
    R₀,₅ R₀,₉ losses

    Results
    -------
    .. epigraph::

        Table 1: Results summary (R₀,₅/R₀,₉ -loss) of all methods. e-c and t-c represent
        electricity-c and traffic-c, respectively. In the 1st and 3rd row, we perform rolling-day
        prediction of 7 days while in the 2nd and 4th row, we directly forecast 7 days ahead.
        TRMF outputs points predictions, so we only report R₀,₅.

    +------+-------------+-------------+------------+-------------+-------------+-------------+
    |      | ARIMA       | ETS         | TRMF       | DeepAR      | DeepState   | Ours        |
    +======+=============+=============+============+=============+=============+=============+
    | e-c₁ | 0.154/0.102 | 0.101/0.077 | 0.084/---- | 0.075/0.040 | 0.083/0.056 | 0.059/0.034 |
    +------+-------------+-------------+------------+-------------+-------------+-------------+
    | e-c₇ | 0.283/0.109 | 0.121/0.101 | 0.087/---- | 0.082/0.053 | 0.085/0.052 | 0.070/0.044 |
    +------+-------------+-------------+------------+-------------+-------------+-------------+
    | t-c₁ | 0.223/0.137 | 0.236/0.148 | 0.186/---- | 0.161/0.099 | 0.167/0.113 | 0.122/0.081 |
    +------+-------------+-------------+------------+-------------+-------------+-------------+
    | t-c₇ | 0.492/0.280 | 0.509/0.529 | 0.202/---- | 0.179/0.105 | 0.168/0.114 | 0.139/0.094 |
    +------+-------------+-------------+------------+-------------+-------------+-------------+

    Fine (-f)

    +--------+----------------+-------------+
    |        | electricity-f₁ | traffic-f₁  |
    +========+================+=============+
    | DeepAR | 0.082/0.063    | 0.230/0.150 |
    +--------+----------------+-------------+
    | Ours   | 0.074/0.042    | 0.139/0.090 |
    +--------+----------------+-------------+
    """


class ElectricityNBEATS:
    r"""NBEATS."""


class ElectricityNHITS:
    r"""NHITS."""
