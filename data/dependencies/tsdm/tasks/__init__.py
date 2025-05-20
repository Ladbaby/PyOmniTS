r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a key to cater to the question of
forecasting horizons.


Idea:

The Pre-Encoder must work in the following way:

- `tuple[TimeTensor] → tuple[TimeTensor]` row-wise!
- `tuple[Tensor] → tuple[Tensor]`

More generally, eligible inputs are:

- `DataFrame`, `TimeTensor`, `tuple[DataFrame]`, `tuple[TimeTensor]`
- Product-types.

Must return a `NamedTuple` that agrees with the original column names!
This allows us to select
"""

__all__ = [
    # Sub- Modules
    "base",
    # Constants
    "Task",
    "TASKS",
    # Classes
    # Tasks
    "ETT_Zhou2021",
    "KIWI_RUNS_TASK",
    "KIWI_FINAL_PRODUCT",
    "ElectricityLim2021",
    "MIMIC_III_DeBrouwer2019",
    "MIMIC_IV_Bilos2021",
    "USHCN_DeBrouwer2019",
    "Kiwi_BioProcessTask",
    "P12"
]


from typing import Final, TypeAlias

from data.dependencies.tsdm.tasks import base
from data.dependencies.tsdm.tasks.base import BaseTask
from data.dependencies.tsdm.tasks.electricity_lim2021 import ElectricityLim2021
from data.dependencies.tsdm.tasks.ett_zhou2021 import ETT_Zhou2021
from data.dependencies.tsdm.tasks.kiwi_bioprocess import Kiwi_BioProcessTask
from data.dependencies.tsdm.tasks.kiwi_final_product import KIWI_FINAL_PRODUCT
from data.dependencies.tsdm.tasks.kiwi_runs_task import KIWI_RUNS_TASK
from data.dependencies.tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
from data.dependencies.tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
from data.dependencies.tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019
from data.dependencies.tsdm.tasks.P12 import Physionet2012

Task: TypeAlias = BaseTask
r"""Type hint for tasks."""

TASKS: Final[dict[str, type[Task]]] = {
    "ETT_Zhou2021": ETT_Zhou2021,
    "KIWI_RUNS_TASK": KIWI_RUNS_TASK,
    "KIWI_FINAL_PRODUCT": KIWI_FINAL_PRODUCT,
    "Kiwi_BioProcessTask": Kiwi_BioProcessTask,
    "ElectricityLim2021": ElectricityLim2021,
    "MIMIC_III_DeBrouwer2019": MIMIC_III_DeBrouwer2019,
    "MIMIC_IV_Bilos": MIMIC_IV_Bilos2021,
    "USHCN_DeBrouwer": USHCN_DeBrouwer2019,
    "P12": Physionet2012
}
r"""Dictionary of all available tasks."""
