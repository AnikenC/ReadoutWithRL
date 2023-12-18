import numpy as numpy
import jax.numpy as jnp

from typing import Optional


def t1_duration(kappa, res_scale_factor, tau_0):
    return (
        -2
        / kappa
        * jnp.log(1 - 1 / res_scale_factor * (1.0 - jnp.exp(-0.5 * kappa * tau_0)))
    )


def t3_duration(kappa, res_scale_factor):
    return -2 / kappa * jnp.log(res_scale_factor / (1.0 + res_scale_factor))
