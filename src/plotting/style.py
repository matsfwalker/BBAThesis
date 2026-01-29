import matplotlib.pyplot as plt
from cycler import cycler, Cycler
from typing import List, Any, Iterator, Sequence, Hashable, Dict
import numpy as np

Colour = Any


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.prop_cycle"] = cycler(color=["#4C72B0", "#55A868", "#C44E52"])


def get_palette(n: int) -> List[np.ndarray]:
    cmap = plt.colormaps["tab10"]
    colours = cmap(np.linspace(0, 1, n))
    return list(colours)


def colour_cycle(palette: Sequence[Colour]) -> Iterator[Colour]:
    while True:
        for c in palette:
            yield c


def colour_map(
    keys: Sequence[Hashable], palette: Sequence[Colour]
) -> Dict[Hashable, Colour]:
    m = len(palette)
    return {k: palette[i % m] for i, k in enumerate(keys)}
