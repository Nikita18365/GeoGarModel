"""Small reproducible synthetic survey for exploring the workbench."""
import numpy as np
import pandas as pd
from .grids import Grid


def demo_survey(seed=42):
    rng = np.random.default_rng(seed)
    east = 10500000 + np.arange(40) * 500.
    north = 6500000 + np.arange(32) * 500.
    xx, yy = np.meshgrid(east, north)
    u, v = xx - east.mean(), yy - north.mean()
    gravity = 15 * np.exp(-(u ** 2 + v ** 2) / 4000 ** 2) + .0002 * u
    magnetic = 200 * np.sin(u / 3000) * np.cos(v / 4000)
    seismic = 1600 + .02 * u - .01 * v
    depth = 1200 + 12 * gravity - .2 * magnetic + .5 * (seismic - 1600)
    table = pd.DataFrame({"X": yy.ravel(), "Y": xx.ravel(), "Gravity": gravity.ravel(),
                          "Magnetic": magnetic.ravel(), "Seismic": seismic.ravel()})
    training = table.iloc[rng.choice(len(table), 320, replace=False)].copy()
    training["Depth"] = depth.ravel()[training.index] + rng.normal(0, 5, len(training))
    return (Grid(east, north, gravity, "DEMO · гравитационное поле", "mGal", "EPSG:20910", "m"),
            training.reset_index(drop=True), table)
