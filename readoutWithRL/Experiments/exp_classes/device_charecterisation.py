import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from qiskit import pulse
from qiskit.tools.monitor import job_monitor

from qiskit.providers.backend import Backend
from qiskit.visualization.pulse_v2 import stylesheet

from scipy.optimize import curve_fit

from typing import Optional

from utils import get_single_qubit_pulses, gaussian_func, sinc_func, acquisition_checker

MHz = 1e6
ns = 1e-9

## Chi Charecterisation


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

        acquisition_checker(job_job)

        return (job_job, details)
