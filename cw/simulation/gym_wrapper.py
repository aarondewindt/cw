from typing import Tuple, Any, Union
from abc import ABC, abstractmethod
from threading import Thread, Semaphore


import gym
import xarray as xr
from gym import spaces
from tqdm.auto import trange

from cw.simulation.module_base import ModuleBase
from cw.simulation.logging import BatchLogger


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
        self.environment_semaphore: Union[Semaphore, None] = None
        self.agent_semaphore: Union[Semaphore, None] = None
        self.last_results: Union[xr.Dataset, Any, None] = None
        self.batch_results: Union[xr.Dataset, None] = None

        self.batch_running = False
        self.is_done = False

    def initialize(self, simulation):
        super().initialize(simulation)

    def simulation_thread_target(self, n_steps, n_episodes, progress_bar) -> None:
        """
        Thread target running simulantion batch.

        :param n_steps: Number of steps per episode.
        :param n_episodes: Number of episodes.
        :param progress_bar: Tqdm progress bar for the batch.
        """
        batch_logger = BatchLogger()
        batch_logger.initialize(self.simulation)
        original_logger = self.simulation.logging
        self.simulation.logging = batch_logger

        try:
            for episode_idx in progress_bar:
                # Break the loop if the batch is not running.
                if not self.batch_running:
                    break

                # Restore the initial simulation states at the start of each episode.
                self.simulation.restore_states()

                # Run simulation
                self.last_results = self.simulation.run(n_steps)

        except KeyboardInterrupt:
            self.batch_running = False
        finally:
            self.environment_semaphore = None
            self.agent_semaphore = None
            self.simulation_thread = None
            self.batch_running = False
            if self.batch_results is None:
                self.batch_results = batch_logger.finish_batch()
            else:
                self.batch_results = xr.concat((self.batch_results, batch_logger.finish_batch()), dim="idx")
            self.simulation.logging = original_logger

    def run_batch(self, n_steps, n_episodes):
        if self.simulation_thread is None:
            self.simulation.stash_states()
            self.environment_semaphore = Semaphore(0)
            self.agent_semaphore = Semaphore(0)
            self.simulation_thread = Thread(target=self.simulation_thread_target,
                                            daemon=True, args=(n_steps, n_episodes, trange(n_episodes)))
            self.batch_running = True
            self.simulation_thread.start()
        else:
            print("Batch already running")

    def end_batch(self):
        if self.simulation_thread is not None:
            self.batch_running = False
            self.simulation.stop()

    def run_step(self):
        self.is_done = False
        self.agent_semaphore.release()
        self.environment_semaphore.acquire()

    def run_end(self):
        self.is_done = True
        self.agent_semaphore.release()
        self.environment_semaphore.acquire()

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        # Make sure we are running a simulantion batch.
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

        # Observe environment.
        observation, reward, info = self.observe(self.is_done)
        del self.s

        # Return observation.
        return observation, reward, self.is_done, info

    @abstractmethod
    def act(self, action: Any):
        pass

    @abstractmethod
    def observe(self, done: bool) -> Tuple[Any, float, dict]:
        pass

    def reset(self):
        self.simulation.stop()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
