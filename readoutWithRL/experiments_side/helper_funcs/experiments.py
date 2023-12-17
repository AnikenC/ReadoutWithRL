import numpy as np
import matplotlib.pyplot as plt

from qiskit import pulse
from qiskit.providers.backend import Backend
from qiskit.circuit import Parameter

from typing import Optional, Union

from helper_funcs.utils import *

MHz = 1e6


class RRFreqSpec:
    def __init__(
        self,
        qubit: int,
        backend: Backend,
        freq_span: Optional[float] = None,
        num_experiments: Optional[float] = None,
        freq_linspace: Optional[float] = None,
        fit_func_name: Optional[str] = "gaussian",
        chi_est: Optional[float] = 0.6e6,
    ):
        super().__init__()
        construct_linspace = freq_span == None and num_experiments == None
        valid_setting = np.logical_xor(freq_linspace == None, construct_linspace)

        if not valid_setting:
            raise ValueError(
                "either freq_linspace must be passed or freq_span + num_experiments must be passed"
            )

        self.qubit = qubit
        self.backend = backend
        self.freq_linspace = freq_linspace
        if not construct_linspace:
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

        return (freq_experiments_g, freq_experiments_e, details)

    def run_analysis(
        self,
        results_arr: np.ndarray,
        custom_label: Optional[str] = "Results",
        chi_est: Optional[float] = None,
        fit_func_name: Optional[str] = None,
    ):
        if chi_est is not None:
            self.chi_est = chi_est
        if (
            results_arr.shape[:2] != (2, len(self.freq_linspace))
            or len(results_arr.shape) != 3
        ):
            raise ValueError(
                "results_arr must be of shape 2 x Linspace Size x Num Shots"
            )

        if fit_func_name is not None:
            if fit_func_name not in self.supported_fit_funcs.keys():
                raise ValueError(
                    f"fit_func must be one of {self.supported_fit_funcs.keys()}"
                )
            self.fit_func = self.supported_fit_funcs[fit_func_name]

        mean_res = np.mean(results_arr, axis=-1)
        abs_res = np.abs(mean_res)
        abs_g, abs_e = abs_res

        p0_g = [7.0, -0.5 * self.chi_est, 5.0e6]
        p0_e = [7.0, 0.5 * self.chi_est, 5.0e6]

        popt_g, pcov_g = curve_fit(self.fit_func, self.freq_linspace, abs_g, p0_g)
        popt_e, pcov_e = curve_fit(self.fit_func, self.freq_linspace, abs_e, p0_e)

        f_g = popt_g[1]
        f_e = popt_e[1]
        cal_freq = 0.5 * (f_g + f_e)
        chi = np.abs(f_g - f_e)

        plt.scatter(
            self.freq_linspace / MHz, abs_res[0], label="g data", c="lightgreen"
        )
        plt.plot(
            self.freq_linspace / MHz,
            gaussian_func(self.freq_linspace, *popt_g),
            label=f"Fit g",
        )
        plt.scatter(self.freq_linspace / MHz, abs_res[1], label=f"e data", c="gold")
        plt.plot(
            self.freq_linspace / MHz,
            gaussian_func(self.freq_linspace, *popt_e),
            label=f"Fit e",
        )
        plt.axvline(
            x=f_g / MHz, color="green", label=f"freq g: {int(f_g/MHz*1e3)/1e3}MHz"
        )
        plt.axvline(
            x=f_e / MHz,
            color="orange",
            label=f"freq e: {int(f_e/MHz*1e3)/1e3}MHz",
        )
        plt.axvline(
            x=cal_freq / MHz,
            color="blue",
            label=f"cal freq: {int(cal_freq/MHz*1e3)/1e3}MHz",
        )
        plt.axvline(x=0.0, color="grey", linestyle="dashed", label="Default Freq")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (A.U.)")
        plt.title(f"Frequency Spec {custom_label} Chi: {int(chi/MHz*1e3)/1e3}MHz")
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.show()


class ACStarkPhoton:
    def __init__(
        self,
        qubit: int,
        backend: Backend,
        freq_linspace: np.ndarray,
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
        super().__init__()
        single_q_dict = get_single_qubit_pulses(qubit, backend)
        self.measure_pulse = single_q_dict["meas pulse"]
        meas_duration = get_closest_multiple_of_16(meas_duration)
        meas_sigma = self.measure_pulse.sigma
        meas_width = meas_duration - 4 * meas_sigma

        supported_modes = ["gaussian_square", "gaussian"]
        if mode not in supported_modes:
            raise ValueError(
                f"input mode: {mode} is not supported, valid modes are {supported_modes}"
            )

        if mode == "gaussian_square":
            self.meas_pulse = pulse.GaussianSquare(
                duration=meas_duration,
                amp=meas_amp,
                sigma=meas_sigma,
                width=meas_width,
            )
        if mode == "gaussian":
            self.meas_pulse = pulse.Gaussian(
                duration=meas_duration,
                amp=meas_amp,
                sigma=meas_sigma,
            )

        self.qubit = qubit
        self.backend = backend
        self.freq_linspace = freq_linspace
        self.meas_amp = meas_amp
        self.qubit_amp = qubit_amp
        self.meas_duration = meas_duration
        self.qubit_duration = qubit_duration
        self.buffer_delay_duration = buffer_delay_duration
        self.meas_delay_sec = meas_delay_sec
        self.qubit_sigma_sec = qubit_sigma_sec
        self.mode = mode
        self.delay_duration_dt = delay_duration_dt

    def get_jobs(self):
        freq = Parameter("freq")
        meas_delay = Parameter("meas delay")

        with pulse.build(
            backend=self.backend,
            default_alignment="sequential",
            name="AC Starks Freq Spec",
        ) as q_freq_spec_sched:
            qubit_chan = pulse.drive_channel(self.qubit)
            meas_chan = pulse.measure_channel(self.qubit)

            drive_duration = get_closest_multiple_of_16(self.qubit_duration)
            drive_sigma = get_closest_multiple_of_16(get_dt_from(self.qubit_sigma_sec))
            delay_dur = get_closest_multiple_of_16(self.delay_duration_dt)

            pulse.shift_frequency(freq, qubit_chan)
            with pulse.align_right():
                pulse.play(self.meas_pulse, meas_chan, name="m pulse")
                pulse.delay(meas_delay, meas_chan, name="m delay")
                pulse.play(
                    pulse.Gaussian(
                        duration=drive_duration,
                        amp=self.qubit_amp,
                        sigma=drive_sigma,
                    ),
                    qubit_chan,
                    name="q tone",
                )
                pulse.delay(delay_dur, qubit_chan)
            pulse.delay(
                get_closest_multiple_of_16(self.buffer_delay_duration),
                meas_chan,
                name="b delay",
            )
            pulse.measure(self.qubit, pulse.MemorySlot(self.qubit))

        if isinstance(meas_delay_sec, float):
            meas_delay_sec = np.array([meas_delay_sec])

        big_exp = []
        for m_delay in meas_delay_sec:
            m_delay = get_closest_multiple_of_16(get_dt_from(m_delay))
            for f in self.freq_linspace:
                big_exp.append(
                    q_freq_spec_sched.assign_parameters(
                        {freq: f, meas_delay: m_delay}, inplace=False
                    )
                )

        return big_exp
