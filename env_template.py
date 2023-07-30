import numpy as np
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from dm_control import viewer


def make_tripod():
    model = mjcf.from_path("tripod_radians.xml")

    return model


class Creature(composer.Entity):
    """A multi-legged creature derived from `composer.Entity`."""

    def _build(self):
        self._model = make_tripod()

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
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


NUM_SUBSTEPS = 1  # The number of physics substeps per control timestep. Was 25


class Task(composer.Task):

    def __init__(self, creature):
        self._creature = creature
        self._arena = floors.Floor()

        self._arena.add_free_entity(self._creature)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

        # Configure initial poses
        self._creature_initial_pose = (0, 0, 1)

        # Configure and enable observables
        self._creature.observables.joint_positions.enabled = True
        self._creature.observables.joint_velocities.enabled = True
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
        self._creature.set_pose(physics, position=(0, 0, 1))

    def get_reward(self, physics):
        return 1


creature = Creature()
task = Task(creature)
env = composer.Environment(task, random_state=np.random.RandomState(42))
viewer.launch(env)
