import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import copy

import chex

from typing import Optional

from envs.single_photon_env import SinglePhotonLangevinReadoutEnv
from envs.photon_env import BatchedPhotonLangevinReadoutEnv


def photon_env_dicts():
    return {
        "single_langevin_env": SinglePhotonLangevinReadoutEnv,
        "photon_langevin_readout_env": BatchedPhotonLangevinReadoutEnv,
    }


def waveform_kappa_chi_stability_tester(
    key: chex.PRNGKey,
    waveform: chex.PRNGKey,
    env_name: str,
    main_env_config: dict,
    error_percentage: Optional[float] = 10.0,
    num_vals: Optional[int] = 5,
):
    reduced_config = copy.deepcopy(main_env_config)
    kappa = reduced_config.pop("kappa")
    chi = reduced_config.pop("chi")

    env_class = photon_env_dicts()[env_name]

    def env_tester(key: chex.PRNGKey, kappa: float, chi: float):
        env = env_class(kappa=kappa, chi=chi, **reduced_config)
        env.shot_noise_std = 0.0

        rng, _rng = jax.random.split(key)
        params = env.default_params
        init_obs, init_state = env.reset(_rng, params)

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, init_state, waveform, params)

        max_pF = state.max_pf
        max_photon = state.max_photon
        photon_time = state.photon_time

        return max_pF, max_photon, photon_time

    jitted_tester = jax.jit(jax.vmap(env_tester))

    error_val = error_percentage / 100.0
    kappas = jnp.linspace(
        kappa * (1.0 - error_val), kappa * (1.0 + error_val), num_vals
    )
    chis = jnp.linspace(chi * (1.0 - error_val), chi * (1.0 + error_val), num_vals)

    kappa_chi_grid = jnp.array(jnp.meshgrid(kappas, chis)).reshape(2, -1).T
    kappa_grid = kappa_chi_grid[:, 0]
    chi_grid = kappa_chi_grid[:, 1]

    rng, _rng = jax.random.split(key)
    rng_tester = jax.random.split(_rng, kappa_chi_grid.shape[0])

    res_pF, res_photon, res_photon_time = jitted_tester(
        rng_tester, kappa_grid, chi_grid
    )

    res_pF = res_pF.reshape(num_vals, -1)
    res_photon = res_photon.reshape(num_vals, -1)
    res_photon_time = res_photon_time.reshape(num_vals, -1)
    return res_pF, res_photon, res_photon_time, kappa_chi_grid, kappas, chis


def plot_learning(
    rewards: jnp.ndarray,
    max_pFs: jnp.ndarray,
    photon_times: jnp.ndarray,
    smoothnesses: jnp.ndarray,
    bandwidths: jnp.ndarray,
):
    """
    Takes in arrays of shape (num_updates, num_envs)
    """
    num_updates = rewards.shape[0]

    mean_rewards = jnp.mean(rewards, axis=-1)
    std_rewards = jnp.std(rewards, axis=-1)

    mean_pFs = jnp.mean(max_pFs, axis=-1)
    std_pFs = jnp.std(max_pFs, axis=-1)

    mean_times = jnp.mean(photon_times, axis=-1)
    std_times = jnp.std(photon_times, axis=-1)

    mean_smoothnesses = jnp.mean(smoothnesses, axis=-1)
    std_smoothnesses = jnp.std(smoothnesses, axis=-1)

    mean_bandwidths = jnp.mean(bandwidths, axis=-1)
    std_bandwidths = jnp.std(bandwidths, axis=-1)

    fig, ax = plt.subplots(5, figsize=(8.0, 16.0))

    updates = jnp.arange(num_updates)

    ax[0].plot(mean_rewards, label="mean rewards")
    ax[0].fill_between(
        updates,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="orange",
        alpha=0.4,
    )
    ax[0].set_xlabel("Updates")
    ax[0].set_ylabel("Mean Batch Reward")
    ax[0].legend()

    ax[1].plot(mean_pFs, label="mean pFs")
    ax[1].fill_between(
        updates,
        mean_pFs - std_pFs,
        mean_pFs + std_pFs,
        color="orange",
        alpha=0.4,
    )
    ax[1].set_xlabel("Updates")
    ax[1].set_ylabel("Mean Batch pF")
    ax[1].legend()

    ax[2].plot(mean_times, label="mean photon times")
    ax[2].fill_between(
        updates,
        mean_times - std_times,
        mean_times + std_times,
        color="orange",
        alpha=0.4,
    )
    ax[2].set_xlabel("Updates")
    ax[2].set_ylabel("Mean Batch Photon Time")
    ax[2].legend()

    ax[3].plot(mean_smoothnesses, label="mean noisiness")
    ax[3].fill_between(
        updates,
        mean_smoothnesses - std_smoothnesses,
        mean_smoothnesses + std_smoothnesses,
        color="orange",
        alpha=0.4,
    )
    ax[3].set_xlabel("Updates")
    ax[3].set_ylabel("Mean Batch Noisiness")
    ax[3].set_yscale("log")
    ax[3].legend()

    ax[4].plot(mean_bandwidths, label="mean bandwidths")
    ax[4].fill_between(
        updates,
        mean_bandwidths - std_bandwidths,
        mean_bandwidths + std_bandwidths,
        color="orange",
        alpha=0.4,
    )
    ax[4].set_xlabel("Updates")
    ax[4].set_ylabel("Mean Batch Bandwidths")
    ax[4].legend()

    plt.show()
