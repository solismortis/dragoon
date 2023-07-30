import collections

import numpy as np
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from dm_control import viewer


PERFECT_Z = 1.1
FILEPATH = "tripod.xml"
# TIME_LIMIT = float("inf")
TIME_LIMIT = 5
NUM_SUBSTEPS = 10  # The number of physics substeps per control
# timestep. The default is 25.


class Creature(composer.Entity):
    def _build(self):
        self._model = mjcf.from_path(FILEPATH)

    def _build_observables(self):
        return CreatureObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def actuators(self):
        return tuple(self._model.find_all('actuator'))


# Add simple observable features for joint angles and velocities.
class CreatureObservables(composer.Observables):

    @composer.observable
    def base_position(self):
        base = self._entity.mjcf_model.find('geom', 'baseg')
        return observable.MJCFFeature('xpos', base)

    @composer.observable
    def base_orientation(self):
        return observable.MJCFFeature('xmat',
                                      self._entity.root_body)[8]

    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)

    @composer.observable
    def touch_sensors(self):
        all_sensors = self._entity.mjcf_model.find_all('sensor')
        return observable.MJCFFeature('sensordata', all_sensors)


class Task(composer.Task):

    def __init__(self, creature):
        self._creature = creature
        self._arena = floors.Floor()
        self._arena._ground_geom.friction = (0.005, 0.005, 0.0001)

        self._arena.add_free_entity(self._creature)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

        # Configure and enable observables
        self._creature.observables.base_position.enabled = True
        self._creature.observables.base_orientation.enabled = True
        self._creature.observables.joint_positions.enabled = True
        self._creature.observables.joint_velocities.enabled = True
        self._creature.observables.touch_sensors.enabled = False
        self._task_observables = {}

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = NUM_SUBSTEPS * 0.002

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        pass

    def initialize_episode(self, physics, random_state):
        self._creature.set_pose(physics, position=(0, 0, 0.3))

    def get_reward(self, physics):
        # Get a reversed reward for distances from the "perfect
        # position" (0, 0, PERFECT_Z)
        x, y, z = physics.named.data.xpos['tripod/base']

        # Get +1 for proper base orientation, -1 for the reversed
        # orientation
        ori = self._creature.observables.base_orientation(physics)

        # Get +1 for a touch sensor touching something
        # sensors = self._creature.observables.touch_sensors(physics)
        # sr = 0
        # if sensors[0] > 0: sr += 1
        # if sensors[1] > 0: sr += 1
        # if sensors[2] > 0: sr += 1

        # Max z_reward is 6
        z_reward = -3*(abs(PERFECT_Z - z))**0.5 + 6

        # Max reward is 7
        reward = -abs(abs(x) + abs(y) + z_reward) + ori * 3

        # print(reward)
        return reward


creature = Creature()
task = Task(creature)
env = composer.Environment(task,
                           random_state=np.random.RandomState(42),
                           time_limit=TIME_LIMIT)
env.reset()


if __name__ == '__main__':
    viewer.launch(env)
