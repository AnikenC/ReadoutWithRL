from typing import Any

import chex
import jax
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
from jax import numpy as jnp
