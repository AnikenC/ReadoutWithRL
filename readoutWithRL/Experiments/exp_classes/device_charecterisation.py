import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from qiskit import pulse
from qiskit.tools.monitor import job_monitor

from qiskit.providers.backend import Backend
from qiskit.visualization.pulse_v2 import stylesheet

from scipy.optimize import curve_fit

from typing import Optional

from utils import (
    get_single_qubit_pulses,
    gaussian_func,
    sinc_func,
    acquisition_checker,
    get_dt_from,
)

MHz = 1e6
ns = 1e-9


class StandardAcqScanExp:
    def __init__(
        self,
        qubit: int,
        backend: Backend,
        acq_latency_dt: int,
        acq_start_dt: int,
        acq_end_dt: int,
        num_acq_exp: int,
        acq_meas_dur_dt: int,
    ):
        super().__init__()
        self.qubit = qubit
        self.backend = backend

        single_q_dict = get_single_qubit_pulses(qubit, backend)

        self.dt = self.backend.configuration().dt

        self.x_pulse = single_q_dict["x pulse"]
        self.meas_pulse = single_q_dict["meas pulse"]
        self.meas_delay_dur = single_q_dict["meas delay"].duration
        self.acq_delay_linspace_ns = np.linspace(
            acq_start_dt * self.dt, acq_end_dt * self.dt, num_acq_exp
        )
        self.acq_latency_dt = acq_latency_dt
        self.acq_meas_dur_dt = acq_meas_dur_dt

    def get_jobs(self):
        exp_g = []
        exp_e = []

        for acq_delay_ns in self.acq_delay_linspace_ns:
            acq_delay_dt = get_dt_from(acq_delay_ns, dt=self.dt)
            acq_delay_ns = np.round(acq_delay_ns / 1e-9)
            with pulse.build(
                backend=self.backend,
                default_alignment="left",
                name=f"Standard Acq Scan G: delay {acq_delay_ns} ns",
            ) as acq_exp_g:
                # qubit_chan = pulse.drive_channel(self.qubit)
                meas_chan = pulse.measure_channel(self.qubit)
                acq_chan = pulse.acquire_channel(self.qubit)

                pulse.delay(self.acq_latency_dt, meas_chan)
                pulse.play(self.meas_pulse, meas_chan)
                pulse.delay(self.meas_delay_dur, meas_chan)
                pulse.delay(acq_delay_dt, acq_chan)
                pulse.acquire(
                    self.acq_meas_dur_dt, acq_chan, pulse.MemorySlot(self.qubit)
                )
            exp_g.append(acq_exp_g)

            with pulse.build(
                backend=self.backend,
                default_alignment="left",
                name=f"Standard Acq Scan E: delay {acq_delay_ns} ns",
            ) as acq_exp_e:
                qubit_chan = pulse.drive_channel(self.qubit)
                meas_chan = pulse.measure_channel(self.qubit)
                acq_chan = pulse.acquire_channel(self.qubit)

                with pulse.align_right():
                    pulse.delay(self.acq_latency_dt, meas_chan)
                    pulse.play(self.x_pulse, qubit_chan)

                pulse.play(self.meas_pulse, meas_chan)
                pulse.delay(self.meas_delay_dur, meas_chan)
                pulse.delay(acq_delay_dt, acq_chan)
                pulse.acquire(
                    self.acq_meas_dur_dt, acq_chan, pulse.MemorySlot(self.qubit)
                )
            exp_e.append(acq_exp_e)

        job_job = exp_g + exp_e

        acquisition_checker(job_job, self.backend)

        details = {
            "Total Experiment Size": len(job_job),
            "Acquisition Latency (ns)": np.round(self.acq_latency_dt * self.dt / 1e-9),
            "Acquisition Start Delay (ns)": np.round(
                self.acq_delay_linspace_ns[0] / 1e-9
            ),
            "Acquisition End Delay (ns)": np.round(
                self.acq_delay_linspace_ns[-1] / 1e-9
            ),
            "Acquisition Duration (ns)": np.round(
                self.acq_meas_dur_dt * self.dt / 1e-9
            ),
        }

        return job_job, details


class StandardMeas:
    def __init__(
        self,
        qubit: int,
        backend: Backend,
    ):
        super().__init__()
        self.qubit = qubit
        self.backend = backend

        single_q_dict = get_single_qubit_pulses(qubit, backend)

        self.x_pulse = single_q_dict["x pulse"]

    def get_jobs(self):
        with pulse.build(
            backend=self.backend, default_alignment="sequential", name="Meas G"
        ) as meas_g:
            pulse.measure(qubits=self.qubit, registers=pulse.MemorySlot(self.qubit))

        with pulse.build(
            backend=self.backend, default_alignment="sequential", name="Meas E"
        ) as meas_e:
            qubit_chan = pulse.drive_channel(self.qubit)
            pulse.play(self.x_pulse, qubit_chan)
            pulse.measure(qubits=self.qubit, registers=pulse.MemorySlot(self.qubit))

        job_job = [meas_g, meas_e]

        acquisition_checker(job_job, self.backend)

        return job_job


class RRFreqSpec:
    def __init__(
        self,
        qubit: int,
        backend: Backend,
        freq_span: Optional[float] = None,
        num_experiments: Optional[int] = None,
        fit_func_name: Optional[str] = "gaussian",
        chi_est: Optional[float] = 0.6e6,
    ):
        super().__init__()
        self.qubit = qubit
        self.backend = backend
        self.freq_linspace = np.linspace(
            -0.5 * freq_span, 0.5 * freq_span, num_experiments
        )

        single_q_dict = get_single_qubit_pulses(qubit, backend)

        self.x_pulse = single_q_dict["x pulse"]
        self.supported_fit_funcs = {"gaussian": gaussian_func, "sinc": sinc_func}
        if fit_func_name not in self.supported_fit_funcs.keys():
            raise ValueError(
                f"only the following fit funcs are supported currently: {self.supported_fit_funcs.keys()}"
            )
        self.fit_func = self.supported_fit_funcs[fit_func_name]

        self.chi_est = chi_est

    def get_jobs(self):
        freq_experiments_g = []
        freq_experiments_e = []

        for freq_shift in self.freq_linspace:
            with pulse.build(
                backend=self.backend,
                default_alignment="sequential",
                name=f"freq spec g, shift: {round(freq_shift/1e6, 3)}MHz",
            ) as freq_spec_g_sched:
                meas_chan = pulse.measure_channel(self.qubit)

                pulse.shift_frequency(freq_shift, meas_chan)
                pulse.measure(self.qubit, pulse.MemorySlot(self.qubit))
            freq_experiments_g.append(freq_spec_g_sched)

            with pulse.build(
                backend=self.backend,
                default_alignment="sequential",
                name=f"freq spec e, shift: {round(freq_shift/1e6, 3)}MHz",
            ) as freq_spec_e_sched:
                qubit_chan = pulse.drive_channel(self.qubit)
                meas_chan = pulse.measure_channel(self.qubit)

                pulse.shift_frequency(freq_shift, meas_chan)
                pulse.play(self.x_pulse, qubit_chan)
                pulse.measure(self.qubit, pulse.MemorySlot(self.qubit))
            freq_experiments_e.append(freq_spec_e_sched)

            details = {
                "Total Experiment Size": len(freq_experiments_g)
                + len(freq_experiments_e),
                "Frequency Step Size (MHz)": round(
                    (self.freq_linspace[1] - self.freq_linspace[0]) / 1e6, 3
                ),
                "Frequency Span (MHz)": round(
                    (self.freq_linspace[-1] - self.freq_linspace[0]) / 1e6, 3
                ),
            }

        job_job = freq_experiments_g + freq_experiments_e

        acquisition_checker(job_job, self.backend)

        return (job_job, details)
