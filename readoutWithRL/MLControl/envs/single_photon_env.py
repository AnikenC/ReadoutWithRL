## Readout Environment based on Quantum Langevin Equation simulation of
## readout resonator photon dynamics
## Constructed only for a Single Evaluation
## And then vmapped appropriately

# Imports for Gymnax Environment
from envs.environment_template import SingleStepEnvironment
from gymnax.environments.environment import EnvParams

# Standard Imports
import jax
import jax.numpy as jnp
from jax import lax, config, vmap, jit
from jax.scipy.special import erf
from jax.scipy.integrate import trapezoid
from jax.nn import relu
from gymnax.environments import spaces
from typing import Tuple, Optional
import chex
from flax import struct

import matplotlib.pyplot as plt

# Imports specific to ODE Simulator
from diffrax import (
    diffeqsolve,
    Tsit5,
    LinearInterpolation,
    ODETerm,
    SaveAt,
    PIDController,
)

config.update("jax_enable_x64", True)


@struct.dataclass
class EnvState:
    """
    Flax Dataclass used to store Dynamic Environment State
    All relevant params that get updated each step should be stored here
    """

    mean_reward: float
    mean_pF: float
    mean_photon: float
    mean_photon_time: float
    mean_smoothness: float
    timestep: int


@struct.dataclass
class EnvParams:
    """
    Flax Dataclass used to store Static Environment Params
    All static env params should be kept here, though they can be equally kept
    in the Jax class as well
    """

    args: chex.Array
    batchsize: int
    t1: float
    ideal_photon: float

    window_length: Optional[int] = 15
    kernel: Optional[chex.Array] = jnp.ones(window_length) / window_length
    gauss_mean: Optional[int] = 0.0
    gauss_std: Optional[int] = 1.0
    small_window: Optional[chex.Array] = jnp.linspace(
        -0.5 * (window_length - 1), 0.5 * (window_length - 1), window_length
    )
    gauss_kernel: Optional[chex.Array] = (
        1
        / (jnp.sqrt(2 * jnp.pi) * gauss_std)
        * jnp.exp(-((small_window - gauss_mean) ** 2) / (2 * gauss_std**2))
    )

    photon_penalty: Optional[float] = 50.0
    bandwidth_penalty: Optional[float] = 5.0
    smoothness_penalty: Optional[float] = 2.0
    bandwidth_scale: Optional[float] = 5.0
    time_penalty: Optional[float] = 50.0

    bandwidth_threshold: Optional[float] = 0.05
    smoothness_threshold: Optional[float] = 1.5

    t0: Optional[float] = 0.0

    num_actions: Optional[int] = 121
    num_sim: Optional[int] = 361

    min_action: Optional[float] = -1.0
    max_action: Optional[float] = 1.0

    min_reward: Optional[float] = -1000.0
    max_reward: Optional[float] = 10.0

    min_separation: Optional[float] = 0.0
    max_separation: Optional[float] = 15.0

    min_bandwidth: Optional[float] = 0.0
    max_bandwidth: Optional[float] = 2.0 * bandwidth_scale

    min_photon: Optional[float] = 0.0
    max_photon: Optional[float] = 50.0

    min_smoothness: Optional[float] = 0.0
    max_smoothness: Optional[float] = 20.0

    dt0: Optional[float] = 1e-3
    max_steps: Optional[int] = 4096
    solver: Optional[Tsit5] = Tsit5()

    # Real and Imaginary Components of Ground and Excited Resonator Coherent State
    y0 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)


class SinglePhotonLangevinReadoutEnv(SingleStepEnvironment):
    """
    Jax Compatible environment for finding the optimal Readout Pulse to be played on a Readout Resonator.
    The Action Space consists of a n_actions long, real-valued pulse that in each episode is generated by
    the RL Agent in a single step. There are no observations for this environment, only a fixed 0 output.
    """

    def __init__(
        self,
        kappa: float,
        chi: float,
        kerr: Optional[float] = 0.002,
        time_coeff: Optional[float] = 10.0,
        snr_coeff: Optional[float] = 10.0,
        smoothness_coeff: Optional[float] = 10.0,
        n0: Optional[float] = 30.0,
        tau_0: Optional[float] = 0.398,
        res_amp_scaling: Optional[float] = 2.3,
        nR: Optional[float] = 0.05,
        snr_scale_factor: Optional[float] = 10.0,
        gamma_I: Optional[float] = 1 / 26,
        photon_gamma: Optional[float] = 1 / 300,
        num_t1: Optional[float] = 8.0,
        init_fid: Optional[float] = 1.0 - 1e-3,
        photon_weight: Optional[float] = 1.0,
    ):
        super().__init__()
        self._kappa = kappa
        self._tau = 1 / kappa
        self._chi = chi
        self._kerr = kerr
        self._init_fid = init_fid
        self._t1 = num_t1 * self._tau
        self._ideal_photon = nR
        self._photon_gamma = photon_gamma
        self._gamma_I = gamma_I
        self.ts_sim = jnp.linspace(0.0, self._t1, 361, dtype=jnp.float64)
        self.ts_action = jnp.linspace(0.0, self._t1, 121, dtype=jnp.float64)
        self.float_dtype = jnp.float32
        self.complex_dtype = jnp.complex64
        self.saveat = SaveAt(ts=self.ts_sim)
        self.stepsize_controller = PIDController(
            rtol=1e-3,
            atol=1e-5,
            pcoeff=0.4,
            icoeff=0.3,
            dcoeff=0.0,
            jump_ts=self.ts_action,
        )
        self.a0 = 0.5 * jnp.sqrt(n0 * (self._kappa**2 + self._chi**2))
        self._two_kappa_index = int(0.2 * (len(self.ts_sim) - 1))
        self.photon_uncertainty = 0.5
        self.mu = res_amp_scaling
        self.min_acq_time = 0.032
        self.precompile()
        self.photon_limit = self.determine_max_photon()
        self.baseline_smoothness = self.get_baseline_smoothness()
        self.freqs_shifted = jnp.fft.fftshift(
            jnp.fft.fftfreq(n=len(self.ts_action), d=self._t1 / len(self.ts_action))
            * self._tau
        )
        self.scaling_factor = snr_scale_factor
        self.ind = 50
        self.pF_factor = snr_coeff
        self.time_factor = time_coeff
        self.smoothness_factor = smoothness_coeff
        self.photon_penalty = 100.0
        self.order_penalty = 100.0
        self.amp_penalty = 10.0
        self.actual_max_photons = n0 * (
            1.0
            - 2.0 * jnp.exp(-0.5 * kappa * tau_0) * jnp.cos(0.5 * chi * tau_0)
            + jnp.exp(-kappa * tau_0)
        )
        self.photon_weight = photon_weight
        self.dt = self._t1 / len(self.ts_sim - 1)

    @property
    def default_params(self) -> EnvParams:
        """
        IMPORTANT Retrieving the Default Env Params
        """
        return EnvParams(
            args=jnp.array(
                [0.5 * self._kappa, self._chi, self._kerr],
                dtype=self.float_dtype,
            ),
            t1=self._t1,
            ideal_photon=self._ideal_photon,
        )

    def determine_max_photon(self):
        """Physics Specific Function"""
        real_action = jnp.ones_like(self.ts_action, dtype=jnp.float64)
        res_drive = self.a0 * real_action
        res_drive = self.drive_smoother(res_drive)
        results = self.calc_results(res_drive)
        results_g = results[:, 0] + 1.0j * results[:, 1]
        max_photon = jnp.max(jnp.abs(results_g) ** 2)
        return max_photon

    def t1(self):
        """Physics Specific Function"""
        return -2 / self._kappa * jnp.log(1.0 - 1 / self.mu)

    def t3(self):
        """Physics Specific Function"""
        return -2 / self._kappa * jnp.log(self.mu / (1.0 + self.mu))

    def dummy_a3r_waveform(
        self,
        t1: Optional[float] = None,
        t2: Optional[float] = None,
        t3: Optional[float] = None,
    ):
        """Physics Specific Function"""
        if t1 is None:
            t1 = self.t1()
        if t2 is None:
            t2 = 0.25
        if t3 is None:
            t3 = self.t3()
        signal = jnp.heaviside(t1 - self.ts_action, 0.0)
        signal += (
            1
            / self.mu
            * (
                jnp.heaviside(t2 - self.ts_action, 0.0)
                - jnp.heaviside(t1 - self.ts_action, 0.0)
            )
        )
        signal -= jnp.heaviside(t2 + t3 - self.ts_action, 0.0) - jnp.heaviside(
            t2 - self.ts_action, 0.0
        )
        return signal

    def get_baseline_smoothness(self):
        """Physics Specific Function"""
        signal = self.dummy_a3r_waveform()
        smoothed_signal = self.drive_smoother(signal)
        smoothness = self.calculate_batch_smoothness(smoothed_signal)
        return smoothness

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, chex.Array, bool, dict]:
        """
        IMPORTANT Perform Single Episode State Transition
        - key is for RNG, needs to be handled properly if used
        - state is the input state, will be modified to produce new state
        - action is an array corresponding to action space shape
        - params is the appropriate Env Params, this argument shouldn't change during training runs

        Returns Observation, New State, Reward, Dones Signal, and Info based on State
        In this particular task, the observation is always fixed, and the Dones is
        always True since its a single-step environment.
        """
        new_timestep = state.timestep + 1

        # Preparing Action for Simulation
        res_drive = self.a0 * action.astype(jnp.float64)
        normalizing_factor = jnp.clip(
            self.mu * self.a0 / jnp.absolute(res_drive),
            0.0,
            1.0,
        )
        res_drive *= normalizing_factor
        res_drive = self.drive_smoother(res_drive)

        # Simulation and Obtaining Reward + Params for New State
        single_result = self.calc_results(res_drive)
        reward, updated_state_array = self.calc_reward_and_state(
            single_result.astype(self.float_dtype), res_drive
        )

        # Setting New State with Updated State Array,
        # New Optimal Action (for logging),
        # and New Timestep
        updated_state = EnvState(*updated_state_array[:-1], new_timestep)
        """
        new_state_return and old_state_return functions are defined to replicate
        if-else functionality for normal state updates. Here I want to retrieve 
        the action that produces the max reward as well as the simulation 
        results due to this action. 
        
        If the best action in the current batch has the highest reward, 
        then new_state_return is called and the current state is returned.

        If the old best action has a higher reward, then old_state_return is
        called, so that the current mean batch statistics is updated, however
        all other state variables are that of the old_state.
        """

        def new_state_return(old_state: EnvState, new_state: EnvState) -> EnvState:
            return new_state

        def old_state_return(old_state: EnvState, new_state: EnvState) -> EnvState:
            updated_old_state = EnvState(
                mean_batch_reward=new_state.mean_batch_reward,
                mean_batch_pF=new_state.mean_batch_pF,
                mean_batch_photon=new_state.mean_batch_photon,
                mean_batch_photon_time=new_state.mean_batch_photon_time,
                mean_batch_smoothness=new_state.mean_batch_smoothness,
                max_batch_reward=old_state.max_batch_reward,
                pF_at_max=old_state.pF_at_max,
                photon_at_max=old_state.photon_at_max,
                photon_time=old_state.photon_time,
                smoothness_at_max=old_state.smoothness_at_max,
                std_batch_reward=new_state.std_batch_reward,
                action_of_max=old_state.action_of_max,
                timestep=new_state.timestep,
            )
            return updated_old_state

        # Condition on whether old_state_return or new_state_return is called
        condition = updated_state.max_batch_reward > state.max_batch_reward
        env_state = jax.lax.cond(
            condition,
            new_state_return,
            old_state_return,
            state,
            updated_state,
        )  # Using jax.lax.cond to obtain the if-else functionality

        done = True
        return (
            lax.stop_gradient(self.get_obs()),
            lax.stop_gradient(env_state),
            reward,
            done,
            lax.stop_gradient(self.get_info(env_state)),
        )

    def drive_smoother(self, res_drive: chex.Array):
        """Physics Specific Function"""
        params = self.default_params
        return jnp.convolve(res_drive, params.gauss_kernel, mode="same")

    def calculate_batch_smoothness(self, batched_action):
        """Physics Specific Function"""
        ts = self.ts_action
        dx = 1.0
        b_first_deriv = jnp.diff(batched_action, axis=-1) / dx
        b_second_deriv = jnp.diff(b_first_deriv, axis=-1) / dx
        integral_val = trapezoid(y=b_second_deriv**2, x=ts[2:], axis=-1)
        return integral_val

    def get_bandwidth(self, batched_drive: chex.Array):
        """Physics Specific Function"""
        params = self.default_params
        fft_vals = jnp.fft.fft(batched_drive, axis=-1)
        fft_shifted = jnp.abs(jnp.fft.fftshift(fft_vals))

        indices = jnp.where(
            fft_shifted > params.bandwidth_threshold * jnp.max(fft_shifted),
            size=params.num_actions,
        )[0]
        min_index = indices[0]
        max_index = jnp.max(indices)
        bandwidth = jnp.array(
            [self.freqs_shifted[max_index] - self.freqs_shifted[min_index]],
            dtype=self.float_dtype,
        )
        return bandwidth

    def extract_values(self, results: chex.Array, action: chex.Array):
        """Physics Specific Function"""
        res_g = results[:, 0] + 1.0j * results[:, 1]
        res_e = results[:, 2] + 1.0j * results[:, 3]

        photon_g = jnp.abs(res_g) ** 2
        photon_e = jnp.abs(res_e) ** 2

        decay_g = -(self._gamma_I + self._photon_gamma * photon_g[1:].T) * self.dt
        decay_g = jnp.vstack((jnp.ones(1), decay_g))
        decay_g = self._init_fid * jnp.cumsum(decay_g, axis=0).T

        photons_combined = jnp.concatenate(
            (photon_g.reshape(-1, 1), photon_e.reshape(-1, 1)), axis=-1
        )
        higher_photons = jnp.max(photons_combined, axis=-1)
        sep = jnp.abs(res_g - res_e)
        fidelity = 0.5 * (1.0 + erf(sep * self.scaling_factor)) * decay_g
        pf = -jnp.log10(1.0 - fidelity)

        max_photons = jnp.max(higher_photons)
        min_photons_ind = jnp.argmin(higher_photons[self.ind :]) + self.ind
        min_photon_time = self.ts_sim[min_photons_ind]

        higher_photons += self.photon_limit * jnp.heaviside(
            self.ts_sim - min_photon_time, 0.0
        )
        closest_time_to_reset_ind = (
            jnp.argmin(jnp.abs(higher_photons[self.ind :] - self.ideal_photon), axis=-1)
            + self.ind
        )
        closest_time_to_reset_action_ind = (
            closest_time_to_reset_ind
            / (len(self.ts_sim) - 1)
            * (len(self.ts_action) - 1)
        )

        closest_time_to_reset_action_ind = closest_time_to_reset_action_ind.astype(
            jnp.int32
        )

        pulse_end_time = self.ts_sim[closest_time_to_reset_ind]
        pulse_reset_val = jnp.abs(action[closest_time_to_reset_action_ind]) + jnp.abs(
            action[0]
        )
        pulse_reset_val /= self.a0 * self.mu

        pulse_reset_photon = 

        pulse_reset_photons = b_higher_photons[
            jnp.arange(self._batchsize), closest_time_to_reset_ind
        ]

        photon_reset_time = pulse_end_times + self.photon_weight * self._tau * jnp.log(
            pulse_reset_photons / self._ideal_photon
        )
        photon_reset_time = jnp.clip(photon_reset_time, a_min=pulse_end_times)

        b_pf *= jnp.heaviside(
            photon_reset_time.reshape(-1, 1) - self.ts_sim.reshape(1, -1), 1.0
        )
        max_pf = jnp.max(b_pf, axis=-1)
        max_pf_times = self.ts_sim[jnp.argmax(b_pf, axis=-1)]

        b_actions_normed = b_actions / (self.mu * self.a0)
        b_smoothness = self.batched_fast_smoothness_calc(b_actions_normed)

        return (
            max_pf,
            max_photons,
            photon_reset_time,
            pulse_end_times,
            max_pf_times,
            b_smoothness,
            b_pf,
            b_higher_photons,
            pulse_reset_vals,
        )

    def calc_results(
        self, res_drive: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Physics Specific Function, Function used for ODE Simulation"""
        params = self.default_params
        control = LinearInterpolation(ts=self.ts_action, ys=res_drive)

        def vector_field(t, y, args):
            res_g_real, res_g_imag, res_e_real, res_e_imag = y
            drive_res = control.evaluate(t)
            kappa_half, chi, kerr = args

            res_g = res_g_real + 1.0j * res_g_imag
            res_e = res_e_real + 1.0j * res_e_imag

            d_res_g = (
                res_g
                * (
                    -kappa_half
                    - 0.5j * chi
                    + 1.0j * kerr * (res_g_real**2 + res_g_imag**2)
                )
                - 1.0j * drive_res
            )
            d_res_e = (
                res_e
                * (
                    -kappa_half
                    - 0.5j * chi
                    + 1.0j * chi
                    + 1.0j * kerr * (res_e_real**2 + res_e_imag**2)
                )
                - 1.0j * drive_res
            )

            return jnp.array(
                [d_res_g.real, d_res_g.imag, d_res_e.real, d_res_e.imag],
                dtype=jnp.float64,
            )

        ode_term = ODETerm(vector_field)

        sol = diffeqsolve(
            terms=ode_term,
            solver=params.solver,
            t0=0.0,
            t1=self._t1,
            dt0=params.dt0,
            y0=params.y0,
            args=params.args,
            saveat=self.saveat,
            stepsize_controller=self.stepsize_controller,
            max_steps=params.max_steps,
        )

        return sol.ys

    def precompile(self):
        """Function called on initialisation to jit + vmap the relevant functions"""
        self.batched_results = jit(vmap(self.calc_results, in_axes=0))
        self.batched_reward_and_state = jit(self.calc_reward_and_state)
        self.batched_smoother = jit(vmap(self.drive_smoother, in_axes=0))
        self.batched_extract_values = jit(self.extract_values)
        self.batched_fast_smoothness_calc = jit(self.calculate_batch_smoothness)

    def calc_reward_and_state(
        self,
        result: chex.Array,
        res_drive: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Function holding Reward Calculation and State Param Calculations"""
        (
            max_pf,
            max_photons,
            photon_reset_time,
            pulse_end_times,
            max_pf_times,
            smoothness_vals,
            b_pf,
            b_higher_photons,
            pulse_reset_vals,
        ) = self.extract_values(result, res_drive)
        # The above function holds physics-specific details

        # Batched Reward is a function of:
        # 1. Max Negative Log Error During the Readout (use Absolute Values)
        # 2. Photon Reset Time
        # 3. Penalty Addition for max_photons (High RELU scaling)
        batched_reward = (
            self.pF_factor * max_pf
            - self.time_factor * photon_reset_time
            - self.smoothness_factor
            * relu((smoothness_vals / self.baseline_smoothness) - 1.0)
            - self.photon_penalty * relu((max_photons / self.actual_max_photons - 1.0))
            - self.order_penalty * (1.0 - jnp.sign(pulse_end_times - max_pf_times))
            - self.amp_penalty * pulse_reset_vals
        )

        # Below code is to retrieve mean params of the batch and the params
        # corresponding to the max reward action (useful for info)
        max_reward = jnp.max(batched_reward)
        max_reward_index = jnp.argmax(max_reward)

        pF_at_max = max_pf[max_reward_index]
        photon_at_max = max_photons[max_reward_index]
        photon_reset_time_at_max = photon_reset_time[max_reward_index]
        smoothness_at_max = smoothness_vals[max_reward_index]

        mean_batch_reward = jnp.mean(batched_reward)
        mean_max_photon = jnp.mean(max_photons)
        mean_batch_pF = jnp.mean(max_pf)
        mean_batch_photon_time = jnp.mean(photon_reset_time)
        mean_smoothness = jnp.mean(smoothness_at_max)

        std_batch_reward = jnp.std(batched_reward)

        state = jnp.array(
            [
                mean_batch_reward,
                mean_batch_pF,
                mean_max_photon,
                mean_batch_photon_time,
                mean_smoothness,
                max_reward,
                pF_at_max,
                photon_at_max,
                photon_reset_time_at_max,
                smoothness_at_max,
                std_batch_reward,
                max_reward_index,
            ],
            dtype=jnp.float64,
        )

        return (batched_reward, state)

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """IMPORTANT Reset Environment, in this case nothing needs to be done
        so default obs and info are returned"""
        # self.precompile()
        state = EnvState(
            mean_batch_reward=0.0,
            mean_batch_pF=0.0,
            mean_batch_photon=0.0,
            mean_batch_photon_time=0.0,
            mean_batch_smoothness=0.0,
            max_batch_reward=-1e5,
            pF_at_max=0.0,
            photon_at_max=0.0,
            photon_time=0.1,
            smoothness_at_max=0.0,
            std_batch_reward=1.0,
            action_of_max=jnp.zeros((params.num_actions,), dtype=self.float_dtype),
            timestep=0,
        )
        return self.get_obs(params), state

    def get_obs(self, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        """IMPORTANT Function to get observation at a given state, as this is a single-step
        episode environment, the observation can be left fixed"""
        return jnp.zeros((self._batchsize,), dtype=jnp.float64)

    def get_info(self, env_state: EnvState) -> dict:
        """IMPORTANT Function to get info for a given input state"""
        return {
            "mean batch reward": env_state.mean_batch_reward,
            "mean batch pF": env_state.mean_batch_pF,
            "mean batch photon": env_state.mean_batch_photon,
            "mean batch photon time": env_state.mean_batch_photon_time,
            "mean batch smoothness": env_state.mean_batch_smoothness,
            "max reward obtained": env_state.max_batch_reward,
            "pF at max": env_state.pF_at_max,
            "photon at max": env_state.photon_at_max,
            "photon time of max": env_state.photon_time,
            "smoothness at max": env_state.smoothness_at_max,
            "std batch reward": env_state.std_batch_reward,
            "action of max": env_state.action_of_max,
            "timestep": env_state.timestep,
        }

    @property
    def name(self) -> str:
        """IMPORTANT name of environment"""
        return "ResonatorReadoutEnv"

    @property
    def num_actions(self, params: Optional[EnvParams] = EnvParams) -> int:
        """IMPORTANT number of actions"""
        return params.num_actions

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """IMPORTANT action space shape"""
        if params is None:
            params = self.default_params

        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(params.num_actions,),
            dtype=jnp.float64,
        )

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """IMPORTANT observation space shape"""
        return spaces.Box(-1.0, 1.0, shape=(1,), dtype=jnp.float64)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """IMPORTANT state space shape"""
        low = jnp.array(
            [
                params.min_reward,
                params.min_separation,
                params.min_photon,
                params.min_reward,
                params.min_separation,
                params.min_photon,
                params.min_smoothness,
                params.min_bandwidth,
                params.t0,
                0.0,
            ],
            dtype=jnp.float64,
        )
        high = jnp.array(
            [
                params.max_reward,
                params.max_separation,
                params.max_photon,
                params.max_reward,
                params.max_separation,
                params.max_photon,
                params.max_smoothness,
                params.max_bandwidth,
                self._t1,
                self._batchsize,
            ],
            dtype=jnp.float64,
        )
        return spaces.Dict(
            {
                "mean batch reward": spaces.Box(low[0], high[0], (), dtype=jnp.float64),
                "mean batch separation": spaces.Box(
                    low[1], high[1], (), dtype=jnp.float64
                ),
                "mean batch photon": spaces.Box(low[2], high[2], (), dtype=jnp.float64),
                "max batch reward": spaces.Box(low[3], high[3], (), dtype=jnp.float64),
                "separation at max": spaces.Box(low[4], high[4], (), dtype=jnp.float64),
                "photon at max": spaces.Box(low[5], high[5], (), dtype=jnp.float64),
                "bandwidth at max": spaces.Box(low[6], high[6], (), dtype=jnp.float64),
                "smoothness at max": spaces.Box(low[7], high[7], (), dtype=jnp.float64),
                "time of max": spaces.Box(low[8], high[8], (), dtype=jnp.float64),
                "index of max": spaces.Box(low[9], high[9], (), dtype=jnp.float64),
            }
        )
