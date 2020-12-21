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
    reset_starting_new_episode = reset | first_step


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
        self.end_semaphore: Union[BinarySemaphore, None] = None

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
        batch_logger = BatchLogger()
        batch_logger.initialize(self.simulation)
        original_logger = self.simulation.logging
        self.simulation.logging = batch_logger

        # Set reset state to not resetting, and processing the first step.
        self.reset_state = ResetState.no_reset_first_step
        try:
            while True:
                # Break the loop if the batch is not running.
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
                    self.reset_state = ResetState.reset_starting_new_episode
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
        if self.simulation_thread is None:
            self.simulation.stash_states()

            # Initial semaphore count of 1 to bound it there.
            self.environment_semaphore = BinarySemaphore(False)
            self.agent_semaphore = BinarySemaphore(False)
            self.reset_semaphore = BinarySemaphore(False)
            self.end_semaphore = BinarySemaphore(False)

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

    def run_step(self):
        if self.reset_state == ResetState.reset_ending_episode:
            # This can be left out if we assume the reset and step functions are called
            # in the same thread. However I know me, and me will come up with some `brilliant`
            # idea at one point. So I might as well add this here so the simulation doesn't
            # end up in a undebuggable freeze.
            self.agent_semaphore.release()
        else:
            with self.is_done_lock:
                self.is_done = False
            self.agent_semaphore.release()

            if self.reset_state == ResetState.reset_starting_new_episode:
                self.reset_semaphore.release()

            self.environment_semaphore.acquire()

    def run_end(self):
        if self.reset_state & ResetState.no_reset:
            with self.is_done_lock:
                self.is_done = True
            self.agent_semaphore.release()
            self.end_semaphore.acquire()

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        # We are running the first step, so change the reset state.
        # And make sure we block on the agent_semaphore later on.
        if self.reset_state == ResetState.no_reset_first_step:
            self.reset_state = ResetState.no_reset_running
            self.agent_semaphore.acquire(False)

        # Make sure we are running a simulation batch.
        if self.simulation_thread is None:
            raise Exception("Simulation batch not running.")

        # States alias
        self.s = self.simulation.states

        # Perform action.
        self.act(action)

        # Release the simulation.
        self.environment_semaphore.release()

        # Lock until the simulation has run the iteration.
        self.agent_semaphore.acquire()

        with self.is_done_lock:
            # Observe environment.
            observation, reward, info = self.observe(self.is_done)
            del self.s

            result = observation, reward, self.is_done, info

            if self.is_done:
                self.end_semaphore.release()
        return result

    @abstractmethod
    def act(self, action: Any):
        pass

    @abstractmethod
    def observe(self, done: bool) -> Tuple[Any, float, dict]:
        pass

    def reset(self):
        # No action have been taken yet on the current episode
        # so there is no need to reset it. Just return the current
        # observation.
        if self.reset_state == ResetState.no_reset_first_step:
            # States alias
            self.s = self.simulation.states
            observation = self.observe(self.is_done)[0]
            del self.s
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
        self.s = self.simulation.states
        observation = self.observe(self.is_done)[0]
        del self.s
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

