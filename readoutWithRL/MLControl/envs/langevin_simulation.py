import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.integrate import trapezoid

import matplotlib.pyplot as plt
from typing import Optional, Union

# Imports specific to ODE Simulator
from diffrax import (
    diffeqsolve,
    Tsit5,
    LinearInterpolation,
    ODETerm,
    SaveAt,
    PIDController,
)

jax.config.update("jax_enable_x64", True)

def run_sim(
    chi: float,
    kappa: float,
    
):
