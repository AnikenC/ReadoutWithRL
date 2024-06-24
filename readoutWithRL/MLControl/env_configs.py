import copy
import jax.numpy as jnp

# Sherbrooke Q1 Sim Config
_sherbrooke_sim_config = {
    "kappa": 14.31,
    "chi": 0.31 * 2.0 * jnp.pi,
    "kerr": 0.00,
    "time_coeff": 1.0,
    "snr_coeff": 20.0,
    "smoothness_coeff": 1.0,
    "smoothness_baseline_scale": 1.0,
    "gauss_kernel_len": 15,
    "gauss_kernel_std": 2.0,
    "bandwidth": 50.0,
    "freq_relative_cutoff": 0.1,
    "bandwidth_coeff": 0.0,
    "n0": 53.8,
    "tau_0": 0.783,
    "res_amp_scaling": 1 / 0.348,
    "nR": 0.05,
    "snr_scale_factor": 1.25,
    "gamma_I": 1 / 362.9,
    "photon_gamma": 1 / 4000,
    "sim_t1": 0.63,
    "init_fid": 1.0,
    "photon_weight": 12.0,
    "standard_fid": 0.99,
    "shot_noise_std": 0.0,
}

# Kyoto Q2 Sim Config
_kyoto_sim_config = {
    "kappa": 10.07,
    "chi": 0.92 * 2.0 * jnp.pi,
    "kerr": 0.002,
    "time_coeff": 2.0,
    "snr_coeff": 20.0,
    "smoothness_coeff": 1.0,
    "smoothness_baseline_scale": 0.5,
    "gauss_kernel_len": 15,
    "gauss_kernel_std": 2.0,
    "bandwidth": 50.0,
    "freq_relative_cutoff": 0.1,
    "bandwidth_coeff": 0.0,
    "n0": 25.5,
    "tau_0": 0.783,
    "res_amp_scaling": 1 / 0.51,
    "nR": 0.1,
    "snr_scale_factor": 0.6,
    "gamma_I": 1 / 286,
    "photon_gamma": 1 / 1200,
    "sim_t1": 0.6,
    "init_fid": 1.0,
    "photon_weight": 8.0,
    "standard_fid": 0.99,
    "shot_noise_std": 0.0,
}


def get_sherbrooke_config():
    return copy.deepcopy(_sherbrooke_sim_config)


def get_kyoto_config():
    return copy.deepcopy(_kyoto_sim_config)
