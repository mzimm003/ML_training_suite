from ml_training_suite.base import ML_Element
from ml_training_suite.registry import Registry

from pettingzoo import AECEnv
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv as PZE
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from pettingzoo.utils.wrappers import BaseWrapper as PZBaseWrapper
from pettingzoo.utils.wrappers import (
    OrderEnforcingWrapper as PZOrderEnforcingWrapper,
    TerminateIllegalWrapper as PZTerminateIllegalWrapper,
    AssertOutOfBoundsWrapper as PZAssertOutOfBoundsWrapper,
    ClipOutOfBoundsWrapper as PZClipOutOfBoundsWrapper,
)

class BaseWrapper(PZBaseWrapper):
    """
    To correct for bug in pettingzoo as of version 1.24.3. Issue is described
    here, https://github.com/Farama-Foundation/PettingZoo/issues/1176. Fix has
    been merged but not released. Until then, this patch suffices to ensure
    the agent selection actually transitions between players in the environment.
    """
    @property
    def agent_selection(self) -> str:
        return self.env.unwrapped.agent_selection

    @agent_selection.setter
    def agent_selection(self, new_val: str):
        self.env.unwrapped.agent_selection = new_val

class OrderEnforcingWrapper(BaseWrapper, PZOrderEnforcingWrapper):
    pass

class TerminateIllegalWrapper(BaseWrapper, PZTerminateIllegalWrapper):
    pass

class AssertOutOfBoundsWrapper(BaseWrapper, PZAssertOutOfBoundsWrapper):
    pass

class ClipOutOfBoundsWrapper(BaseWrapper, PZClipOutOfBoundsWrapper):
    pass

class PettingZooEnv(PZE):
    def __init__(self, env:AECEnv):
        MultiAgentEnv().__init__()
        self.env:AECEnv = env
        env.reset()

        self._agent_ids = set(self.env.agents)

        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space

    def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = [next(iter(self._agent_ids))]
            # agent_ids = self._agent_ids
        return {aid: self._observation_space(aid).sample() for aid in agent_ids}

    def observation_space(self, agent_id = None):
        if agent_id is None:
            agent_id = next(iter(self._agent_ids))
        return self._observation_space(agent_id)

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = [next(iter(self._agent_ids))]
            # agent_ids = self._agent_ids
        return {aid: self._action_space(aid).sample() for aid in agent_ids}

    def action_space(self, agent_id = None):
        if agent_id is None:
            agent_id = next(iter(self._agent_ids))
        return self._action_space(agent_id)

#Necessary to avoid instantiating environment outside of worker, causing shared parameters
#between what should be independent environments.
def env_creator(env, **kwargs):
    env = env(**kwargs)
    return env

class Environment(PettingZooEnv, ML_Element):
    registry = Registry()

    def __init__(
            self,
            env,
            render_mode=None) -> None:
        super().__init__(env_creator(env, render_mode=render_mode))
        register_env(
            self.__class__.__name__,
            lambda config: PettingZooEnv(env_creator(env, render_mode=render_mode)))
    
    @property
    def render_mode(self) -> str:
        return self.env.render_mode
    
    @property
    def agent_selection(self) -> str:
        return self.env.agent_selection
    
    @property
    def agents(self) -> str:
        return self.env.agents
    
    def render(self):
        return self.env.render()

    def getName(self):
        return self.__class__.__name__
    
    @staticmethod
    def simulate_move(self, board, action, player):
        pass
    
    @staticmethod
    def simulate_observation(self, board, input):
        pass