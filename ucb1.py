import math


def ind_max(x):
    m = max(x)
    return x.index(m)


class UCB1():
    #def __init__(self, counts):
     #   self.counts = counts
      #  #self.q_values = q_values
      #  return

    def __init__(self, n_actions):
        self.counts = [0 for col in range(n_actions)]
        self.q_values = [0.0 for col in range(n_actions)]
        return

    #def initialize(self, n_actions):
    #    self.counts = [0 for col in range(n_actions)]
    #    self.q_values = [0.0 for col in range(n_actions)]
    #    return

    def return_bonus(self):
        n_actions = len(self.counts)
        for action_idx in range(n_actions):
            if self.counts[action_idx] == 0:
                return action_idx

        bonus = [0.0 for action_idx in range(n_actions)]
        total_counts = sum(self.counts)
        for action_idx in range(n_actions):
            bonus[action_idx] = math.sqrt((2 * math.log(total_counts)) / float(self.counts[action_idx]))
            #ucb_q_values[action_idx] = self.q_values[action_idx] + bonus
        return bonus

    def select_action_idx(self):
        n_actions = len(self.counts)
        for action_idx in range(n_actions):
            if self.counts[action_idx] == 0:
                return action_idx

        ucb_q_values = [0.0 for action_idx in range(n_actions)]
        total_counts = sum(self.counts)
        for action_idx in range(n_actions):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[action_idx]))
            ucb_q_values[action_idx] = self.q_values[action_idx] + bonus
        return ind_max(ucb_q_values)

    def update(self, chosen_action_idx):
        self.counts[chosen_action_idx] = self.counts[chosen_action_idx] + 1
        #n = self.counts[chosen_action_idx]

        #q_value = self.q_values[chosen_action_idx]
        #new_q_value = ((n - 1) / float(n)) * q_value + (1 / float(n)) * reward
        #self.q_values[chosen_action_idx] = new_q_value
        return