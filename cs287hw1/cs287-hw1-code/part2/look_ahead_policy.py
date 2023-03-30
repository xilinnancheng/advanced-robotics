import numpy as np
from gym import spaces


class LookAheadPolicy(object):
    """
    Look ahead policy

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
    * self.horizon (int): Horizon for the look ahead policy

    * act_dim (int): Dimension of the state space

    * value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states
    * env (Env):
                - vec_set_state(states): vectorized (multiple environments in parallel) version of reseting the
                environment to a state for a batch of states.
                - vec_step(actions): vectorized (multiple environments in parallel) version of stepping through the
                environment for a batch of actions. Returns the next observations, rewards, dones signals, env infos
                (last not used).
    """

    def __init__(self,
                 env,
                 value_fun,
                 horizon,
                 ):
        self.env = env
        self.discount = env.discount
        self._value_fun = value_fun
        self.horizon = horizon

    def get_action(self, state):
        """
        Get the best action by doing look ahead, covering actions for the specified horizon.
        HINT: use np.meshgrid to compute all the possible action sequences.
        :param state:
        :return: best_action (int)
           """
        assert isinstance(self.env.action_space, spaces.Discrete)
        """ INSERT YOUR CODE HERE"""
        act_dim = self.env.action_space.n
        action_array = []
        for i in range(self.horizon):
            action_array.append([x for x in range(act_dim)])
        action_sequences = np.array(np.meshgrid(
            *action_array)).T.reshape(-1, self.horizon).T
        return action_sequences[0, np.argmax(self.get_returns(state, action_sequences))]

    def get_returns(self, state, actions):
        """
        :param state: current state of the policy
        :param actions: array of actions of shape [horizon, num_acts]
        :return: returns for the specified horizon + self.discount ^ H value_fun
        HINT: Make sure to take the discounting and done into acount!
        """
        assert self.env.vectorized
        """ INSERT YOUR CODE HERE"""
        num_action_sequence = actions.shape[1]
        returns = np.zeros(num_action_sequence)
        self.env.vec_set_state(np.tile(state, (num_action_sequence)))
        for i in range(self.horizon):
            next_state, rewards, done, env_infos = self.env.vec_step(
                actions[i])
            self.env.vec_set_state(next_state)
            returns += self.discount ** i * rewards

        returns += self.discount ** self.horizon * \
            self._value_fun.get_values(next_state)
        return returns

    def update(self, actions):
        pass
