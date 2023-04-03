import numpy as np
from utils.utils import DiscretizeWrapper
from collections.abc import Iterable


class Discretize(DiscretizeWrapper):
    """
    Discretize class: Discretizes a continous gym environment


    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
        * self.state_points (np.ndarray): grid that contains the real values of the discretization

        * self.obs_n (int): number of discrete points

        * self.transitions (np.ndarray): transition matrix of size (S+1, A, S+1). The last state corresponds to the sink
                                         state
        * self.rewards (np.ndarray): reward matrix of size (S+1, A, S+1). The last state corresponds to the sink state

        * self.get_id_from_coordinates(coordinate_vector) returns the id of the coordinate_vector

        * self.get_state_from_id(id_s): get the continuous state associated to that state id

        * self.get_action_from_id(id_a): get the contiouns action associated to that action id

        * env.set_state(s): resets the environment to the continous state s

        * env.step(a): applies the action a to the environment. Returns next_state, reward, done, env_infos. The
                            last element is not used.
    """

    def vec_get_discrete_state_from_cont_state(self, cont_state):
        """
        Get discrete state from continuous state.
            * self.mode (str): specifies if the discretization is to the nearest-neighbour (nn) or n-linear (linear).

        :param cont_state (np.ndarray): of shape env.observation_space.shape
        :return: A tuple of (states, probs). states is a np.ndarray of shape (1,) if mode=='nn'
                and (2 ^ obs_dim,) if mode=='linear'. probs is the probabability of going to such states,
                it has the same size than states.
        """
        """INSERT YOUR CODE HERE"""
        cont_state = np.expand_dims(cont_state, axis=-1)
        if self.mode == 'nn':
            closest_coor = np.argmin(
                np.abs(self.state_points - cont_state), axis=-1)
            id_s = self.get_id_from_coordinates(closest_coor)
            if isinstance(id_s, Iterable):
                states = id_s
                probs = np.tile(np.array([1]), len(id_s))
            else:
                states = [id_s]
                probs = [1]

        elif self.mode == 'linear':
            cont_state_tmp = cont_state
            for index in range(cont_state_tmp.shape[0]):
                for i in range(len(cont_state_tmp[index])):
                    cont_state_tmp[index, i] = max(
                        cont_state_tmp[index, i], self.state_points[i, 0] + 1e-3)
                    cont_state_tmp[index, i] = min(
                        cont_state_tmp[index, i], self.state_points[i, self.state_points.shape[1] - 1] - 1e-3)

            upper_coor = np.argmax(
                self.state_points > cont_state_tmp, axis=-1)
            lower_coor = upper_coor - 1
            cs = np.column_stack((lower_coor, upper_coor))

            print(upper_coor.shape)
            raise NotImplementedError

            upper_state = np.expand_dims(
                self.state_points[np.arange(self.obs_dims), upper_coor], axis=-1)
            lower_state = np.expand_dims(
                self.state_points[np.arange(self.obs_dims), lower_coor], axis=-1)

            upper_p = (cont_state_tmp - lower_state) / \
                (upper_state - lower_state)
            lower_p = (upper_state - cont_state_tmp) / \
                (upper_state - lower_state)
            ps = np.column_stack((lower_p, upper_p))

            # id_list.append(self.get_id_from_coordinates(np.array(
            #     np.meshgrid(*cs)).T.reshape(-1, self.obs_dims)))
            # prob_list.append(np.prod(np.array(np.meshgrid(
            #     *ps)).T.reshape(-1, self.obs_dims), axis=1))

            # states = np.vstack(id_list)
            # probs = np.vstack(prob_list)
        else:
            raise NotImplementedError
        return states, probs

    def vec_add_transitions(self, id_s, id_a):
        """
        Populates transition and reward matrix (self.transition and self.reward)
        :param id_s (int): discrete index of the the state
        :param id_a (int): discrete index of the the action

        """
        env = self._wrapped_env
        obs_n = self.obs_n
        curr_states = self.get_state_from_id(id_s)
        curr_actions = self.get_action_from_id(id_a)

        """INSERT YOUR CODE HERE"""
        env.vec_set_state(curr_states)
        next_states, rewards, dones, env_infos = env.vec_step(curr_actions)
        next_state_ids, probs = self.vec_get_discrete_state_from_cont_state(
            next_states)

        for index in range(id_s.shape[0]):
            if dones[index]:
                self.transitions[id_s[index], id_a[index], obs_n] = 1
                self.rewards[id_s[index], id_a[index], obs_n] = rewards[index]
            else:
                if self.mode == 'nn':
                    self.transitions[id_s[index],
                                     id_a[index], next_state_ids[index]] = probs[index]
                    self.rewards[id_s[index], id_a[index],
                                 next_state_ids[index]] = rewards[index]
                elif self.mode == 'linear':
                    for i in range(len(next_state_ids[index])):
                        self.transitions[id_s[index], id_a[index],
                                         next_state_ids[index, i]] = probs[index, i]
                        self.rewards[id_s[index], id_a[index],
                                     next_state_ids[index, i]] = rewards[index]
                else:
                    raise NotImplementedError

    def add_done_transitions(self):
        """
        Populates transition and reward matrix for the sink state (self.transition and self.reward). The sink state
        corresponds to the last state (self.obs_n or -1).
        """
        """INSERT YOUR CODE HERE"""
        self.transitions[-1, :, -1] = 1
        self.rewards[-1, :, -1] = 0
