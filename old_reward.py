"""Overcomplicated standing reward"""


def get_reward(self, physics):
    # TODO: Rewards for base stability when close to perfect
    #  coords?
    # Get a reversed reward for distances from the "perfect
    # position" (0, 0, PERFECT_Z)
    x, y, z = physics.named.data.xpos['unnamed_model/base_b']

    # Get +1 for proper base orientation, -1 for the reversed
    # orientation
    ori = self._creature.observables.base_orientation(physics)

    # Max z_reward is 6
    z_reward = -3 * (abs(PERFECT_Z - z)) ** 0.5 + 6

    # Max reward is 7
    reward = -abs(abs(x) + abs(y) + z_reward) + ori * 3
    # print(reward)
    return reward
