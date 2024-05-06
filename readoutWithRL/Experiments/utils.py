import numpy as np
from scipy.optimize import curve_fit

from qiskit import pulse
from qiskit.providers.backend import Backend
from qiskit.circuit import Parameter
from qiskit.result import Result
from qiskit.visualization.pulse_v2 import stylesheet

from scipy.special import erf

from typing import Optional, Union


import math


def classify(point: complex, mean_a, mean_b):
    """Classify the given state as |0> or |1>."""

    def distance(a, b):
        return math.sqrt(
            (np.real(a) - np.real(b)) ** 2 + (np.imag(a) - np.imag(b)) ** 2
        )

    return int(distance(point, mean_b) < distance(point, mean_a))


def classify_results(res_array, mean_g, mean_e):
    """
    Array inputs need to be of shape (num_exp, num_shots)
    """

    prob_0_arr = np.zeros(res_array.shape[0])
    prob_1_arr = np.zeros(res_array.shape[0])

    for ind, arr_obj in enumerate(res_array):
        g = 0
        e = 0

        for result in arr_obj:
            res = classify(result, mean_g, mean_e)
            if res == 0:
                g += 1
            if res == 1:
                e += 1

        prob_0_arr[ind] = g / (g + e)
        prob_1_arr[ind] = e / (g + e)

    return prob_0_arr, prob_1_arr


def get_fidelity(array_g, array_e):
    """
    Array inputs need to be of shape (num_exp, num_shots)
    """
    means_g = np.mean(array_g, axis=-1)
    means_e = np.mean(array_e, axis=-1)
    overall_means = 0.5 * (means_g + means_e)
    overall_means = overall_means.reshape(-1, 1)

    array_g -= overall_means
    array_e -= overall_means

    means_g = np.mean(array_g, axis=-1)
    means_e = np.mean(array_e, axis=-1)

    angle_dev = 0.5 * (np.angle(means_g) + np.angle(means_e)) - 0.5 * np.pi
    angle_dev = angle_dev.reshape(-1, 1)

    array_g *= np.exp(-1.0j * angle_dev)
    array_e *= np.exp(-1.0j * angle_dev)

    array_g = array_g.real
    array_e = array_e.real

    means_g = np.mean(array_g, axis=-1)
    means_e = np.mean(array_e, axis=-1)
    sep = np.abs(means_g - means_e)

    std_g = np.std(array_g, axis=-1)
    std_e = np.std(array_e, axis=-1)
    avg_std = 0.5 * (std_g + std_e)

    sep_fidelity = 0.5 * (1.0 + erf(sep / (2 * np.sqrt(2) * avg_std)))

    num_shots = array_g.shape[-1]

    fidelity_arr = np.zeros(len(means_g))

    for ind, (g_mean, e_mean) in enumerate(zip(means_g, means_e)):
        g_gnd = 0
        g_exc = 0

        e_gnd = 0
        e_exc = 0

        for result in array_g[ind]:
            res = classify(result, g_mean, e_mean)
            if res == 0:
                g_gnd += 1
            if res == 1:
                g_exc += 1

        for result in array_e[ind]:
            res = classify(result, g_mean, e_mean)
            if res == 0:
                e_gnd += 1
            if res == 1:
                e_exc += 1

        fidelity = 1.0 - 0.5 * (g_exc + e_gnd) / num_shots
        fidelity_arr[ind] = fidelity
    return fidelity_arr, sep_fidelity


def flatten(inp_list):
    arr = np.array(inp_list).reshape(-1)
    return arr.tolist()


def gaussian_func(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))


def gaussian_func_with_offset(x, a, b, c, d):
    return a * np.exp(-((x - b) ** 2) / (2 * c**2)) + d


def sinc_func(x, a, b, c):
    return a * np.sinc((x - b) / c)


def get_closest_multiple_of(value, base_number):
    return int(value + base_number / 2) - (int(value + base_number / 2) % base_number)


# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return get_closest_multiple_of(num, 16)


# Convert seconds to dt
def get_dt_from(sec, dt: Optional[float] = 1 / 4.5 * 1.0e-9):
    return get_closest_multiple_of(sec / dt, 16)


def convert_arr_to_dt(arr, dt: Optional[float] = 1 / 4.5e9):
    return np.round(arr / (16 * dt)) * 16 * dt


def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit


def fit_gaussian_with_offset(x_values, y_values, init_params):
    """
    Init Params should have 4 Values (Amplitude, Mean, STD, Offset)
    """

    params, cov = fit_function(
        x_values, y_values, gaussian_func_with_offset, init_params
    )
    fit_res = gaussian_func_with_offset(x_values, *params)
    return params, fit_res


def acquisition_checker(job: list, backend):
    acq_duration_list = []
    all_duration_list = []
    for schedule in job:
        instructions_array = np.array(schedule.instructions)
        ops_array = instructions_array[:, 1]
        for op in ops_array:
            if isinstance(op, pulse.Acquire):
                acq_duration_list.append(op.duration)
            if isinstance(op, pulse.Play):
                all_duration_list.append(op.duration)

    acq_duration_array = np.array(acq_duration_list)
    all_duration_array = np.array(all_duration_list)

    alignment_context = backend.configuration().timing_constraints["pulse_alignment"]

    if not (all_duration_array % alignment_context == 0).all():
        raise ValueError(
            "At least one Delay or Play Instruction has a duration that is not divisible by 16"
        )

    if not (all_duration_array != 0).all():
        raise ValueError("At least one Delay or Play Instruction has a duration of 0")

    if len(acq_duration_array) != len(job):
        raise ValueError(
            "There are less/more than one acquisition instructions per circuit in the job"
        )

    if not (acq_duration_array == acq_duration_array[0]).all():
        raise ValueError("All Acquisition Durations are not identical")

    if acq_duration_array[0] % alignment_context != 0:
        raise ValueError(
            "All Acquisition Durations must be positive integer multiples of 16"
        )


def get_single_qubit_pulses(qubit: int, backend: Backend) -> dict:
    """
    Returns a Dictionary of Single Qubit + Resonator Operations containing:
        X Pulse
        SX Pulse
        Measure Pulse
        Measure Delay
    """
    instruction_schedule_map = backend.defaults().instruction_schedule_map
    measure_instructions = np.array(
        instruction_schedule_map.get("measure", qubits=[qubit]).instructions
    )
    measure_pulse = measure_instructions[-2, 1].pulse
    measure_delay = measure_instructions[-1, 1]

    x_instructions = np.array(
        instruction_schedule_map.get("x", qubits=[qubit]).instructions
    )
    x_pulse = x_instructions[0, 1].pulse

    sx_instructions = np.array(
        instruction_schedule_map.get("sx", qubits=[qubit]).instructions
    )
    sx_pulse = sx_instructions[0, 1].pulse

    single_qubit_rr_dict = {
        "x pulse": x_pulse,
        "sx pulse": sx_pulse,
        "meas pulse": measure_pulse,
        "meas delay": measure_delay,
    }
    return single_qubit_rr_dict


def get_results_arr(result: Result, qubit: int, scale_factor: Optional[float] = 1e-7):
    big_list = []

    for i in range(len(result.to_dict()["results"])):
        big_list.append(result.get_memory(i)[:, qubit] * scale_factor)

    big_arr = np.array(big_list)
    return big_arr
