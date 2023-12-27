import numpy as numpy
import jax.numpy as jnp
from jax.scipy.special import erf

import matplotlib.pyplot as plt

from envs.photon_langevin_env import BatchedPhotonLangevinReadoutEnv

from typing import Optional


def t1_duration(kappa, res_scale_factor, tau_0):
    return (
        -2
        / kappa
        * jnp.log(1 - 1 / res_scale_factor * (1.0 - jnp.exp(-0.5 * kappa * tau_0)))
    )


def t3_duration(kappa, res_scale_factor):
    return -2 / kappa * jnp.log(res_scale_factor / (1.0 + res_scale_factor))


def pF(x):
    return -jnp.log10(1 - x)


def get_fidelity_curves(
    kappa,
    chi,
    kerr,
    gamma,
    time_coeff,
    snr_coeff,
    smoothness_coeff,
    rough_max_photons,
    actual_max_photons,
    rough_max_amp_scaled,
    ideal_photon,
    scaling_factor,
    gamma_I,
    num_t1,
    photon_gamma,
    init_fid,
    tau_0,
    plot_curves: Optional[bool] = True,
    action: Optional[jnp.ndarray] = None,
):
    batch_size = 1
    ind = 0

    env = BatchedPhotonLangevinReadoutEnv(
        kappa=kappa,
        chi=chi,
        batchsize=batch_size,
        kerr=kerr,
        gamma=gamma,
        time_coeff=time_coeff,
        snr_coeff=snr_coeff,
        smoothness_coeff=smoothness_coeff,
        rough_max_photons=rough_max_photons,
        actual_max_photons=actual_max_photons,
        rough_max_amp_scaled=rough_max_amp_scaled,
        ideal_photon=ideal_photon,
        scaling_factor=scaling_factor,
        gamma_I=gamma_I,
        num_t1=num_t1,
        photon_gamma=photon_gamma,
        init_fid=init_fid,
    )

    ts = env.ts_sim

    batch_action = jnp.tile(action, (batch_size, 1))
    batched_results = env.batched_results(batch_action)
    (
        max_pf,
        max_photons,
        photon_reset_time,
        smoothness_val,
        b_pf,
        b_higher_photons,
    ) = env.batched_extract_values(batched_results, batch_action)
    real_pf = b_pf[0]

    t1 = t1_duration(kappa, rough_max_amp_scaled, tau_0)
    t3 = t3_duration(kappa, rough_max_amp_scaled)
    decay_factor = 1.0 - jnp.exp(-0.5 * kappa * tau_0)

    u = rough_max_amp_scaled
    N0_u = rough_max_photons * u**2
    N0_f = rough_max_photons * decay_factor**2

    integrated_decay_t1 = (
        (gamma + photon_gamma * N0_u) * t1
        + 4 * photon_gamma * N0_u / kappa * (jnp.exp(-0.5 * kappa * t1) - 1.0)
        - photon_gamma * N0_u / kappa * (jnp.exp(-kappa * t1) - 1.0)
    )
    fidelity_high_t1 = init_fid * jnp.exp(-integrated_decay_t1)

    fidelity_i2 = fidelity_high_t1 * jnp.exp(-(gamma + photon_gamma * N0_f) * (ts - t1))
    b1 = (
        u
        * (
            chi / kappa * (1.0 - jnp.exp(-0.5 * kappa * t1))
            - 0.5 * chi * t1 * jnp.exp(-0.5 * kappa * t1)
        )
        / decay_factor
    )

    separation_i2 = (
        2
        * jnp.sqrt(N0_f)
        * (
            b1 * jnp.exp(-0.5 * kappa * (ts - t1))
            + chi / kappa * (1.0 - jnp.exp(-0.5 * kappa * (ts - t1)))
        )
    )
    opt_fidelity = 0.5 * (1 + erf(scaling_factor * separation_i2)) * fidelity_i2

    integrated_decay_ts = (
        (gamma + photon_gamma * rough_max_photons) * ts
        + 4
        * photon_gamma
        * rough_max_photons
        / kappa
        * (jnp.exp(-0.5 * kappa * ts) - 1.0)
        - photon_gamma * rough_max_photons / kappa * (jnp.exp(-kappa * ts) - 1.0)
    )
    fidelity_normal = init_fid * jnp.exp(-integrated_decay_ts)
    separation_normal = (
        2
        * jnp.sqrt(rough_max_photons)
        * (
            chi / kappa * (1.0 - jnp.exp(-0.5 * kappa * ts))
            - 0.5 * chi * ts * jnp.exp(-0.5 * kappa * ts)
        )
    )
    exp_fidelity = 0.5 * (1 + erf(scaling_factor * separation_normal)) * fidelity_normal

    integrated_high_decay_ts = (
        (gamma + photon_gamma * N0_u) * ts
        + 4 * photon_gamma * N0_u / kappa * (jnp.exp(-0.5 * kappa * ts) - 1.0)
        - photon_gamma * N0_u / kappa * (jnp.exp(-kappa * ts) - 1.0)
    )
    fidelity_high = init_fid * jnp.exp(-integrated_high_decay_ts)
    separation_high = (
        2
        * jnp.sqrt(N0_u)
        * (
            chi / kappa * (1.0 - jnp.exp(-0.5 * kappa * ts))
            - 0.5 * chi * ts * jnp.exp(-0.5 * kappa * ts)
        )
    )
    high_fidelity = 0.5 * (1 + erf(scaling_factor * separation_high)) * fidelity_high

    peak_opt_fidelity = jnp.max(opt_fidelity)
    peak_opt_fidelity_time = ts[jnp.argmax(opt_fidelity)]

    peak_exp_fidelity = jnp.max(exp_fidelity)
    peak_exp_fidelity_time = ts[jnp.argmax(exp_fidelity)]

    peak_real_fidelity = jnp.max(real_pf)
    peak_real_fidelity_time = ts[jnp.argmax(real_pf)]

    peak_high_fidelity = jnp.max(high_fidelity)
    peak_high_fidelity_time = ts[jnp.argmax(high_fidelity)]

    max_fidelity = init_fid * jnp.exp(-gamma * ts)

    if plot_curves:
        fig, ax = plt.subplots(1, figsize=(10.0, 8.0))

        ax.plot(ts, pF(exp_fidelity), label="exp pF", color="red")
        ax.plot(ts, pF(opt_fidelity), label="opt pF", color="green")
        ax.plot(ts, pF(max_fidelity), label="max pF", color="blue")
        ax.plot(ts, pF(high_fidelity), label="high pF", color="purple")
        ax.plot(ts, real_pf, label="action pF", color="orange")
        ax.axvline(x=t1, color="grey", linestyle="dashed", label="t1")
        ax.axvline(
            x=peak_exp_fidelity_time,
            color="red",
            linestyle="dashed",
            label=f"exp t2: {int(peak_exp_fidelity_time * 1e3) / 1e3}us",
        )
        ax.axvline(
            x=peak_high_fidelity_time,
            color="purple",
            linestyle="dashed",
            label=f"high t2: {int(peak_high_fidelity_time * 1e3) / 1e3}us",
        )
        ax.axvline(
            x=peak_opt_fidelity_time,
            color="green",
            linestyle="dashed",
            label=f"opt t2: {int(peak_opt_fidelity_time * 1e3) / 1e3}us",
        )
        ax.axvline(
            x=peak_real_fidelity_time,
            color="orange",
            linestyle="dashed",
            label=f"real t2: {int(peak_real_fidelity_time * 1e3) / 1e3}us",
        )
        ax.axhline(
            y=peak_exp_fidelity,
            color="red",
            linestyle="dashed",
            label=f"exp max fid: {int(peak_exp_fidelity * 1e3) / 1e3}",
        )
        ax.axhline(
            y=peak_high_fidelity,
            color="purple",
            linestyle="dashed",
            label=f"high max fid: {int(peak_high_fidelity * 1e3) / 1e3}",
        )
        ax.axhline(
            y=peak_opt_fidelity,
            color="green",
            linestyle="dashed",
            label=f"opt max fid: {int(peak_opt_fidelity * 1e3) / 1e3}",
        )
        ax.axhline(
            y=peak_real_fidelity,
            color="orange",
            linestyle="dashed",
            label=f"real max fid: {int(peak_real_fidelity * 1e3) / 1e3}",
        )
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("pF")
        ax.set_title("pF vs Time (us)")
        ax.legend()

        print(f"Kappa: {kappa} MHz")
        print(f"1/kappa: {1000 * 1/kappa}ns")
        print(f"Normal Measurement Duration: {1000 * tau_0}ns")
        print(f"High Amp Duration (T1): {1000 * t1}ns")

        plt.show()

    return t1, peak_opt_fidelity_time, t3, photon_reset_time


def generate_A3R_waveform(kappa, chi, tau_0, num_t1, t1, t2, t3, N0, u):
    ts = jnp.linspace(0.0, num_t1 / kappa, 121)
    amp_2 = 1 / u * (1.0 - jnp.exp(-0.5 * kappa * tau_0))
    t2 -= 2 / 3 * t3

    A3R_action = jnp.heaviside(t1 - ts, 0.0)
    A3R_action += amp_2 * (jnp.heaviside(t2 - ts, 0.0) - jnp.heaviside(t1 - ts, 0.0))
    A3R_action -= jnp.heaviside(t2 + t3 - ts, 0.0) - jnp.heaviside(t2 - ts, 0.0)
    A3R_action *= jnp.heaviside(t2 + t3 - ts, 0.0)

    A3R_action *= 0.5 * u * jnp.sqrt(N0 * (kappa**2 + chi**2))

    return A3R_action, ts
