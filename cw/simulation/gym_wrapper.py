from typing import Tuple, Any, Union
from abc import ABC, abstractmethod
from threading import Thread, RLock, Condition
from enum import Flag, auto
from concurrent.futures import Future

import gym
import xarray as xr
from gym import spaces
from tqdm.auto import trange
from tqdm import tqdm

from cw.simulation.module_base import ModuleBase
from cw.simulation.logging import BatchLogger
from cw.synchronization import BinarySemaphore


class ResetState(Flag):
    no_reset = auto()
    reset = auto()
    running = auto()
    first_step = auto()

    no_reset_running = no_reset | running
    no_reset_first_step = no_reset | first_step
    reset_ending_episode = reset | running
    reset_first_step = reset | first_step


class GymEnvironment(ModuleBase, gym.Env):
    metadata = {'render.modes': []}

    def __init__(self,
                 required_states=None,
                 target_time_step=None):
        super().__init__(
            is_discreet=True,
            target_time_step=target_time_step,
            required_states=required_states,
        )

        self.simulation_thread: Union[Thread, None] = None

        self.environment_semaphore: Union[BinarySemaphore, None] = None
        self.agent_semaphore: Union[BinarySemaphore, None] = None
        self.reset_semaphore: Union[BinarySemaphore, None] = None

        self.last_results: Union[xr.Dataset, Any, None] = None
        self.last_batch_results: Union[xr.Dataset, None] = None

        self.simulation_result_futures = []

        self.simulation_result_futures_lock = RLock()
        self.is_done_lock = RLock()
        self.step_lock = RLock()

        self.thread_running = False
        self.reset_state = ResetState.no_reset_first_step
        self.is_done = False

    def initialize(self, simulation):
        super().initialize(simulation)

    def simulation_thread_target(self, n_steps) -> None:
        """
        Thread target running simulantion batch.

        :param n_steps: Number of steps per episode.
        """
        # Set a batch logger on the simulation.
        batch_logger = BatchLogger()
        batch_logger.initialize(self.simulation)
        original_logger = self.simulation.logging
        self.simulation.logging = batch_logger

        # Set reset state to not resetting, and processing the first step.
        self.reset_state = ResetState.no_reset_first_step
        try:
            while True:
                # Break the loop if the thread is not supposed to be running anymore.
                if not self.thread_running:
                    break

                # Restore the initial simulation states at the start of each episode.
                self.simulation.restore_states()

                # Run simulation
                self.last_results = self.simulation.run(n_steps)

                with self.simulation_result_futures_lock:
                    for future in self.simulation_result_futures:
                        if not future.done():
                            future.set_result(self.last_results)
                    self.simulation_result_futures = []

                # If resetting we need to run the first iteration up the agent and
                # let the reset function return an observation of the first step.
                if self.reset_state & ResetState.reset:
                    # Set reset state to resetting and first iteration.
                    self.reset_state = ResetState.reset_first_step
                else:
                    # Set reset state to not resetting and first step.
                    self.reset_state = ResetState.no_reset_first_step

        except KeyboardInterrupt:
            self.thread_running = False
        finally:
            self.environment_semaphore = None
            self.agent_semaphore = None
            self.reset_semaphore = None
            self.simulation_thread = None
            self.thread_running = False
            self.last_batch_results = batch_logger.finish_batch()
            self.simulation.logging = original_logger

    def start_simulation_thread(self, n_steps):
        # Don't start simulation thread if it's already running.
        if self.simulation_thread is None:
            self.simulation.stash_states()

            # Initialize binary Semaphores to False to they are blocked by default.
            self.environment_semaphore = BinarySemaphore(False)
            self.agent_semaphore = BinarySemaphore(False)
            self.reset_semaphore = BinarySemaphore(False)

            # Create and start thread.
            self.simulation_thread = Thread(target=self.simulation_thread_target,
                                            daemon=True, args=(n_steps,))
            self.thread_running = True
            self.simulation_thread.start()
        else:
            print("Simulation thread already running")

    def end_simulation_thread(self):
        if self.simulation_thread is not None:
            self.thread_running = False
            self.simulation.stop()

    def run_step(self, is_last):
        if self.reset_state == ResetState.reset_ending_episode:
            return

        with self.is_done_lock:
            self.is_done = is_last

        # Run the environment step function.
        self.environment_step(is_last)

        self.agent_semaphore.release()

        # Release the reset function if this is the first step
        # after a reset.
        if self.reset_state == ResetState.reset_first_step:
            self.reset_semaphore.release()

        self.environment_semaphore.acquire()

    def run_end(self):
        pass

    def environment_step(self, is_last):
        pass

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        if self.reset_state & ResetState.first_step:
            # We are running the first step, so change the reset state.
            # And make sure we block on the agent_semaphore later on.
            self.reset_state = ResetState.no_reset_running
            self.agent_semaphore.acquire(False)

        # Make sure we are running a simulation batch.
        if self.simulation_thread is None:
            raise Exception("Simulation thread not running.")

        # Perform action.
        self.act(action)

        # Release the simulation.
        self.environment_semaphore.release()

        # Lock until the simulation has run the iteration.
        self.agent_semaphore.acquire()

        with self.is_done_lock:
            # Observe environment.
            observation, reward, info = self.observe(self.is_done)

            if self.is_done:
                self.environment_semaphore.release()

            return observation, reward, self.is_done, info

    @abstractmethod
    def act(self, action: Any):
        pass

    @abstractmethod
    def observe(self, done: bool) -> Tuple[Any, float, dict]:
        pass

    def reset(self):
        if self.reset_state == ResetState.no_reset_first_step:
            # No actions have been taken yet on the current episode
            # so there is no need to reset it. Just return the current
            # observation.
            observation = self.observe(self.is_done)[0]
            return observation

        # Set the state to reset and still running (aka, ending episode)
        # Stop simulation
        # Release the environment semaphore to let the simulation finish.
        self.reset_state = ResetState.reset_ending_episode
        self.simulation.stop()
        self.environment_semaphore.release()

        # Wait for the new episode to start.
        self.reset_semaphore.acquire()

        # Return observation.
        observation = self.observe(self.is_done)[0]
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def create_result_future(self):
        future = Future()
        with self.simulation_result_futures_lock:
            self.simulation_result_futures.append(future)
        return future
