import numpy as np
from scipy.optimize import curve_fit

from qiskit import pulse
from qiskit.providers.backend import Backend
from qiskit.circuit import Parameter
from qiskit.result import Result

from typing import Optional, Union


def gaussian_func(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))


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


def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit


def acquisition_checker(job: list):
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

    if not (all_duration_array % 16 == 0).all():
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

    if acq_duration_array[0] % 16 != 0:
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
