import numpy as np
from qiskit import pulse
from qiskit.providers.backend import Backend
from typing import Optional, Union
from qiskit.circuit import Parameter

from helper_funcs.utils import (
    get_closest_multiple_of_16,
    get_dt_from,
    get_single_qubit_pulses,
)


def rr_freq_spec(
    qubit: int,
    backend: Backend,
    freq_span: Optional[float] = None,
    num_experiments: Optional[float] = None,
    freq_linspace: Optional[float] = None,
):
    construct_linspace = freq_span == None and num_experiments == None
    valid_setting = np.logical_xor(freq_linspace == None, construct_linspace)

    if not valid_setting:
        raise ValueError(
            "either freq_linspace must be passed or freq_span + num_experiments must be passed"
        )

    if not construct_linspace:
        freq_linspace = np.linspace(-0.5 * freq_span, 0.5 * freq_span, num_experiments)

    single_q_dict = get_single_qubit_pulses(qubit, backend)
    x_pulse = single_q_dict["x pulse"]

    freq_experiments_g = []
    freq_experiments_e = []

    for freq_shift in freq_linspace:
        with pulse.build(
            backend=backend,
            default_alignment="sequential",
            name=f"freq spec g, shift: {round(freq_shift/1e6, 3)}MHz",
        ) as freq_spec_g_sched:
            meas_chan = pulse.measure_channel(qubit)

            pulse.shift_frequency(freq_shift, meas_chan)
            pulse.measure(qubit, pulse.MemorySlot(qubit))
        freq_experiments_g.append(freq_spec_g_sched)

        with pulse.build(
            backend=backend,
            default_alignment="sequential",
            name=f"freq spec e, shift: {round(freq_shift/1e6, 3)}MHz",
        ) as freq_spec_e_sched:
            qubit_chan = pulse.drive_channel(qubit)
            meas_chan = pulse.measure_channel(qubit)

            pulse.shift_frequency(freq_shift, meas_chan)
            pulse.play(x_pulse, qubit_chan)
            pulse.measure(qubit, pulse.MemorySlot(qubit))
        freq_experiments_e.append(freq_spec_e_sched)

        details = {
            "Total Experiment Size": len(freq_experiments_g) + len(freq_experiments_e),
            "Frequency Step Size (MHz)": round(
                (freq_linspace[1] - freq_linspace[0]) / 1e6, 3
            ),
            "Frequency Span (MHz)": round(
                (freq_linspace[-1] - freq_linspace[0]) / 1e6, 3
            ),
        }

    return (freq_experiments_g, freq_experiments_e, details)


def rr_freq_spec_analysis(data: np.array):
    """
    Data should be of shape num_experiments x num_shots
    """
    if len(data.shape) != 2:
        raise ValueError(
            "data must be a two dimensional array of num_experiments x num_shots"
        )

    freq_abs_data = np.abs(np.mean(data, axis=-1))

    return freq_abs_data


def integrated_fidelity_experiment(qubit: int, backend: Backend):
    ge_experiment = []
    qnd_experiment = []

    single_q_dict = get_single_qubit_pulses(qubit, backend)
    x_pulse = single_q_dict["x pulse"]
    meas_pulse = single_q_dict["meas pulse"]
    meas_delay = single_q_dict["meas delay"]

    with pulse.build(
        backend=backend, default_alignment="sequential", name="meas g"
    ) as meas_g_sched:
        pulse.measure(qubit, pulse.MemorySlot(qubit))
    ge_experiment.append(meas_g_sched)

    with pulse.build(
        backend=backend, default_alignment="sequential", name="meas e"
    ) as meas_e_sched:
        qubit_chan = pulse.drive_channel(qubit)

        pulse.play(x_pulse, qubit_chan)
        pulse.measure(qubit, pulse.MemorySlot(qubit))
    ge_experiment.append(meas_e_sched)

    with pulse.build(
        backend=backend, default_alignment="sequential", name="qnd g"
    ) as qnd_g_sched:
        meas_chan = pulse.measure_channel(qubit)

        pulse.play(meas_pulse, meas_chan)
        pulse.delay(meas_delay.duration, meas_chan)
        pulse.measure(qubit, pulse.MemorySlot(qubit))
    qnd_experiment.append(qnd_g_sched)

    with pulse.build(
        backend=backend, default_alignment="sequential", name="qnd e"
    ) as qnd_e_sched:
        qubit_chan = pulse.drive_channel(qubit)

        pulse.play(x_pulse, qubit_chan)
        pulse.play(meas_pulse, meas_chan)
        pulse.delay(meas_delay.duration, meas_chan)
        pulse.measure(qubit, pulse.MemorySlot(qubit))
    qnd_experiment.append(qnd_e_sched)

    return ge_experiment, qnd_experiment


def general_ramsey_t2_experiment(
    qubit: int,
    backend: Backend,
    freq_detuning: float,
    num_periods: int,
    points_per_period: int,
    meas_block: Optional[pulse.ScheduleBlock] = None,
    buffer_duration: Optional[int] = 0,
    delay_duration_sec: Optional[Union[float, np.array]] = 0.0,
    inp_linspace: Optional[np.array] = None,
):
    if inp_linspace is not None:
        ramsey_t2_linspace = inp_linspace
    if inp_linspace is None:
        ramsey_t2_linspace = np.linspace(
            0.0, num_periods / freq_detuning, num_periods * points_per_period + 1
        )
    big_experiments = []

    if isinstance(delay_duration_sec, float):
        delay_duration_sec = np.array([delay_duration_sec])

    single_q_dict = get_single_qubit_pulses(qubit, backend)
    sx_pulse = single_q_dict["sx pulse"]

    dt = backend.configuration().dt

    buffer_duration = get_closest_multiple_of_16(buffer_duration)

    for delay_dur_sec in delay_duration_sec:
        delay_dur_dt = get_closest_multiple_of_16(get_dt_from(delay_dur_sec))
        ramsey_t2_experiments = []
        for ramsey_delay_sec in ramsey_t2_linspace:
            ramsey_delay_dur = get_closest_multiple_of_16(
                get_dt_from(ramsey_delay_sec, dt)
            )
            with pulse.build(
                backend=backend,
                default_alignment="sequential",
                name=f"long ramsey t2, delay: {round(ramsey_delay_dur * dt * 1e9, 1)}ns",
            ) as ramsey_t2_sched:
                qubit_chan = pulse.drive_channel(qubit)
                meas_chan = pulse.measure_channel(qubit)
                if meas_block is not None:
                    pulse.call(meas_block)
                pulse.delay(
                    delay_dur_dt,
                    meas_chan,
                    name=f"d delay {round(delay_dur_dt*dt*1e9)}ns",
                )
                if freq_detuning != 0.0:
                    pulse.shift_frequency(freq_detuning, qubit_chan)
                pulse.play(sx_pulse, qubit_chan)
                pulse.delay(ramsey_delay_dur, qubit_chan)
                pulse.play(sx_pulse, qubit_chan)
                pulse.delay(
                    buffer_duration,
                    meas_chan,
                    name=f"b delay {round(buffer_duration*dt*1e9)}ns",
                )
                pulse.measure(qubit, pulse.MemorySlot(qubit))
            ramsey_t2_experiments.append(ramsey_t2_sched)

        big_experiments.append(ramsey_t2_experiments)

    return big_experiments, ramsey_t2_linspace


def general_ac_stark_photon_experiment(
    qubit: int,
    backend: Backend,
    freq_linspace: np.array,
    meas_amp: float,
    qubit_amp: float,
    meas_duration: int,
    qubit_duration: int,
    buffer_delay_duration: int,
    meas_delay_sec,
    qubit_sigma_sec: Optional[float] = 15 * 1e-9,
):
    single_q_dict = get_single_qubit_pulses(qubit, backend)
    measure_pulse = single_q_dict["meas pulse"]

    freq = Parameter("freq")
    meas_delay = Parameter("meas delay")

    with pulse.build(
        backend=backend, default_alignment="sequential", name="Qubit Freq Spec"
    ) as q_freq_spec_sched:
        qubit_chan = pulse.drive_channel(qubit)
        meas_chan = pulse.measure_channel(qubit)

        meas_duration = get_closest_multiple_of_16(meas_duration)
        meas_sigma = measure_pulse.sigma
        meas_width = meas_duration - 4 * meas_sigma

        drive_duration = get_closest_multiple_of_16(qubit_duration)
        drive_sigma = get_closest_multiple_of_16(get_dt_from(qubit_sigma_sec))

        pulse.shift_frequency(freq, qubit_chan)
        pulse.play(
            pulse.GaussianSquare(
                duration=meas_duration,
                amp=meas_amp,
                sigma=meas_sigma,
                width=meas_width,
                angle=measure_pulse.angle,
            ),
            meas_chan,
            name="m pulse",
        )
        pulse.delay(meas_delay, meas_chan, name="m delay")
        pulse.play(
            pulse.Gaussian(
                duration=drive_duration,
                amp=qubit_amp,
                sigma=drive_sigma,
            ),
            qubit_chan,
            name="q tone",
        )
        pulse.delay(
            get_closest_multiple_of_16(buffer_delay_duration), meas_chan, name="b delay"
        )
        pulse.measure(qubit, pulse.MemorySlot(qubit))

    if isinstance(meas_delay_sec, float):
        meas_delay_sec = np.array([meas_delay_sec])

    big_exp = []
    for m_delay in meas_delay_sec:
        m_delay = get_closest_multiple_of_16(get_dt_from(m_delay))
        for f in freq_linspace:
            big_exp.append(
                q_freq_spec_sched.assign_parameters(
                    {freq: f, meas_delay: m_delay}, inplace=False
                )
            )

    return big_exp


def improved_ac_stark_photon_experiment(
    qubit: int,
    backend: Backend,
    freq_linspace: np.array,
    meas_amp: float,
    qubit_amp: float,
    meas_duration: int,
    qubit_duration: int,
    buffer_delay_duration: int,
    meas_delay_sec,
    qubit_sigma_sec: Optional[float] = 15 * 1e-9,
    mode: Optional[str] = "gaussian_square",
    delay_duration_dt: Optional[int] = 128,
):
    single_q_dict = get_single_qubit_pulses(qubit, backend)
    measure_pulse = single_q_dict["meas pulse"]

    freq = Parameter("freq")
    meas_delay = Parameter("meas delay")

    meas_duration = get_closest_multiple_of_16(meas_duration)
    meas_sigma = measure_pulse.sigma
    meas_width = meas_duration - 4 * meas_sigma

    if mode == "gaussian_square":
        meas_pulse = pulse.GaussianSquare(
            duration=meas_duration,
            amp=meas_amp,
            sigma=meas_sigma,
            width=meas_width,
            angle=measure_pulse.angle,
        )
    if mode == "rectangular":
        meas_pulse = pulse.Constant(
            duration=meas_duration, amp=meas_amp, angle=measure_pulse.angle
        )

    with pulse.build(
        backend=backend, default_alignment="sequential", name="Qubit Freq Spec"
    ) as q_freq_spec_sched:
        qubit_chan = pulse.drive_channel(qubit)
        meas_chan = pulse.measure_channel(qubit)

        drive_duration = get_closest_multiple_of_16(qubit_duration)
        drive_sigma = get_closest_multiple_of_16(get_dt_from(qubit_sigma_sec))
        delay_dur = get_closest_multiple_of_16(delay_duration_dt)

        pulse.shift_frequency(freq, qubit_chan)
        with pulse.align_right():
            pulse.play(meas_pulse, meas_chan, name="m pulse")
            pulse.delay(meas_delay, meas_chan, name="m delay")
            pulse.play(
                pulse.Gaussian(
                    duration=drive_duration,
                    amp=qubit_amp,
                    sigma=drive_sigma,
                ),
                qubit_chan,
                name="q tone",
            )
            pulse.delay(delay_dur, qubit_chan)
        pulse.delay(
            get_closest_multiple_of_16(buffer_delay_duration), meas_chan, name="b delay"
        )
        pulse.measure(qubit, pulse.MemorySlot(qubit))

    if isinstance(meas_delay_sec, float):
        meas_delay_sec = np.array([meas_delay_sec])

    big_exp = []
    for m_delay in meas_delay_sec:
        m_delay = get_closest_multiple_of_16(get_dt_from(m_delay))
        for f in freq_linspace:
            big_exp.append(
                q_freq_spec_sched.assign_parameters(
                    {freq: f, meas_delay: m_delay}, inplace=False
                )
            )

    return big_exp


def qubit_t1_exp(
    qubit: int,
    backend,
    min_delay: Optional[float] = 0.0,
    max_delay: Optional[float] = 100.0 * 1e-6,
    num_exp: Optional[int] = 101,
):
    single_q_dict = get_single_qubit_pulses(qubit, backend)
    x_pulse = single_q_dict["x pulse"]
    dt = backend.configuration().dt

    delay_linspace = np.linspace(min_delay, max_delay, num_exp)

    t1_decay_exp = []

    for t1_delay in delay_linspace:
        t1_delay_dt = get_closest_multiple_of_16(get_dt_from(t1_delay))
        with pulse.build(
            backend=backend,
            default_alignment="sequential",
            name=f"Qubit T1, Delay: {int(t1_delay_dt * dt/1e-6 * 1e3) / 1e3}us",
        ) as t1_sched:
            qubit_chan = pulse.drive_channel(qubit)

            pulse.play(x_pulse, qubit_chan)
            pulse.delay(t1_delay_dt, qubit_chan)
            pulse.measure(qubit, pulse.RegisterSlot(qubit))
        t1_decay_exp.append(t1_sched)

    details = {
        "Total Experiment Size": len(t1_decay_exp),
        "Frequency Step Size (us)": round(
            (delay_linspace[1] - delay_linspace[0]) / 1e-6, 3
        ),
        "Frequency Span (us)": round(
            (delay_linspace[-1] - delay_linspace[0]) / 1e-6, 3
        ),
    }

    return t1_decay_exp, details
