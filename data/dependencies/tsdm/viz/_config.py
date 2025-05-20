r"""Initialize the plotting module.

Enable LaTeX rendering by default, if installed.
"""

__all__ = [
    # CONSTANTS
    "USE_TEX",
]

import warnings
from typing import Final

import matplotlib

USE_TEX: Final[bool] = matplotlib.checkdep_usetex(True)
r"""Whether to use LaTeX rendering."""


if USE_TEX:
    try:
        matplotlib.use("pgf")
    except ValueError:
        warnings.warn("matplotlib: pgf backend not available")

    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "pgf.texsystem": r"lualatex",
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage{fontspec}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage[utf8x]{inputenc}",
                    r"\usepackage{amsmath}",
                    r"\usepackage{amsfonts}",
                    r"\usepackage{amssymb}",
                    r"\usepackage{unicode-math}",
                ]
            ),
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage{amsfonts}",
                    r"\usepackage{amssymb}",
                ]
            ),
            # "mathtext.fontset": "stix",
            # "font.family": "STIXGeneral",
            # "svg.fonttype": "none",
        }
    )
else:
    warnings.warn("matplotlib: no LaTeX rendering")
