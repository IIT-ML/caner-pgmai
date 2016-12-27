import numpy as np


class GibbsSampling:
    def __init__(self, seed=0):
        self.random_state = np.random.RandomState(seed=seed)

    def reset_random_state(self, seed=0):
        self.rs = np.random.RandomState(seed=seed)

    @staticmethod
    def get_joint_distribution(var_list, beta0, beta, sigma):
        n_var = len(var_list)
        mu = np.zeros((n_var))
        covariance_mat = np.zeros((n_var, n_var))
        for var in var_list:
            mu[var] = beta0[var] + np.dot(beta[var], mu)
            covariance_mat[var] = np.dot(beta[var], covariance_mat)
            covariance_mat[:, var] = covariance_mat[var]
            covariance_mat[var, var] = sigma[var] + np.dot(beta[var], covariance_mat[var])
        return mu, covariance_mat

    @staticmethod
    def find_markov_blanket(self, current_var, parents):
        mb = parents[current_var][:]
        for rand_var in range(len(parents)):
            current_parent_list = parents[rand_var][:]
            if current_var in current_parent_list:
                current_parent_list.remove(current_var)
                mb += current_parent_list
                mb.append(rand_var)
        return mb

    @staticmethod
    def get_conditional_distribution(self, mu, covariance_mat):
        # last variable of the vector and the matrix is the child, all others are its parents.
        beta0 = mu[-1] - np.dot(np.dot(covariance_mat[-1, :-1], np.linalg.inv(covariance_mat[:-1][:, :-1])), mu[:-1])
        beta = np.dot(covariance_mat[-1, :-1], np.linalg.inv(covariance_mat[:-1][:, :-1]))
        sigma = covariance_mat[-1, -1] - np.dot(
            np.dot(covariance_mat[-1, :-1], np.linalg.inv(covariance_mat[:-1][:, :-1])),
            covariance_mat[:-1, -1])
        return beta0, beta, sigma

    def sample_gibbs_temporal(self, var_list, beta0, beta, sigma, n_sample, obs_vars=None):
        if not obs_vars:
            obs_vars = dict()
        sample = list()
        instance = np.zeros((n_var))
        for i in range(n_sample):
            for var in var_list:
                if var in obs_vars.keys():
                    instance[var] = obs_vars[var]
                else:
                    instance[var] = self.random_state.normal(beta0[var] + np.dot(beta[var], instance), sigma[var] ** .5)
            sample.append(list(instance))
        sample = np.vstack(sample)
        return sample