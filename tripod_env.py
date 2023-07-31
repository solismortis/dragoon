"""Run to test the env manually. Be sure to remove time limit. If
you don't see episodic returns, you forgot to add a time limit."""

import numpy as np
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from dm_control import viewer

TASK = 'stand'  # stand or walk
PERFECT_Z = 1.1  # Z coord where we want our agent to be
TIME_LIMIT = 3
if __name__ == '__main__':  # If not imported
    TIME_LIMIT = float("inf")
N_SUBSTEPS = 10  # The number of physics substeps per control. The
# default is 25

ASSETS_DIR = 'assets'
TURRET = False  # For pacifists
N_LEGS = 3  # More than 6 is not recommended for this model
RANDOM_STARTS = False

random_state = np.random.RandomState(42)




class Leg:
    def __init__(self):
        self.model = mjcf.RootElement()

        # Defaults
        self.model.default.joint.type = 'hinge'
        self.model.default.joint.damping = 16
        self.model.default.motor.ctrlrange = (-.7, .7)
        self.model.default.motor.ctrllimited = True

        # Assets
        part1_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/part1.stl")
        part2_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/part2.stl")
        part3_1_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/part3_1.stl")
        part3_2_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/part3_2.stl")
        part4_1_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/part4_1.stl")
        part4_2_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/part4_2.stl")
        part4_3_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/part4_3.stl")

        rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])

        # Part 1
        part1_b = self.model.worldbody.add("body")
        pos = (0, 0, -.1)
        part1_g = part1_b.add("geom", type="mesh", rgba=rgba,
                              mesh=part1_a, pos=pos)
        part1_j = part1_b.add("joint", pos=pos,
                              limited=True, range=(-.8727, .8727))
        self.model.actuator.add('motor', gear=(200, 0, 0),
                                joint=part1_j)

        # Part 2
        part2_b = part1_b.add("body")
        pos = (0, 0, -.204)
        part2_g = part2_b.add("geom", type="mesh", rgba=rgba,
                              mesh=part2_a, pos=pos)
        part2_j = part2_b.add("joint", pos=pos,
                              limited=True, axis=(0, 1, 0),
                              range=(-.261799, 1.5708))
        self.model.actuator.add('motor', gear=(200, 0, 0),
                                joint=part2_j)

        # Part 3
        part3_b = part2_b.add("body")
        pos = (0, 0, -.204)
        part3_1_g = part3_b.add("geom", type="mesh", rgba=rgba,
                                mesh=part3_1_a, pos=pos)
        part3_2_g = part3_b.add("geom", type="mesh", rgba=rgba,
                                mesh=part3_2_a, pos=pos)
        part3_j = part3_b.add("joint", pos=pos, axis=(1, 0, 0),
            limited=True, range=(-0.17, 0.17))
        self.model.actuator.add('motor', gear=(200, 0, 0),
                                joint=part3_j)

        # Part 4
        part4_b = part3_b.add("body")
        pos = (.438, 0, -.202)
        part4_1_g = part4_b.add("geom", type="mesh", rgba=rgba,
                                mesh=part4_1_a, pos=pos)
        part4_2_g = part4_b.add("geom", type="mesh", rgba=rgba,
                                mesh=part4_2_a, pos=pos)
        part4_3_g = part4_b.add("geom", type="mesh", rgba=rgba,
                                mesh=part4_3_a, pos=pos)
        part4_j = part4_b.add("joint", pos=pos, limited=True,
            axis=(0, 1, 0), range=(-0.872665, 2.26893))
        self.model.actuator.add('motor', gear=(200, 0, 0),
                                joint=part4_j)


class Creature(composer.Entity):

    def make_creature(self):
        """Constructs a creature with `N_LEGS` legs."""

        self.model = mjcf.RootElement()

        # Defaults
        self.model.default.joint.type = 'hinge'
        self.model.default.joint.damping = 16
        self.model.default.motor.ctrlrange = (-.7, .7)
        self.model.default.motor.ctrllimited = True

        # Assets
        base_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/base.stl")
        battery_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/battery.stl")
        turret_base_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/turret_base.stl")
        gun_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/gun.stl")
        ammo_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/ammo.stl")
        cams_a = self.model.asset.add("mesh",
            file=f"{ASSETS_DIR}/cams.stl")

        # Torso
        self.base_b = self.model.worldbody.add("body", name="base_b")
        self.base_g = self.base_b.add("geom", type="mesh",
            mesh=base_a, density="200")
        self.battery_g = self.base_b.add("geom", type="mesh",
            mesh=battery_a)

        if TURRET:
            rgba = (0.3, 0.32, 0.3, 1)
            self.turret_b = self.model.worldbody.add("body")
            self.turret_base_g = self.turret_b.add("geom",
                type="mesh", mesh=turret_base_a, rgba=rgba)
            self.turret_base_j = self.turret_b.add("joint")
            self.model.actuator.add('motor', gear=(200, 0, 0),
                                    joint=self.turret_base_j)

            self.gun_set_b = self.turret_b.add("body")
            gun_set_pos = (-0.029, -0.055, 0.187)
            self.gun_g = self.gun_set_b.add("geom", type="mesh",
                mesh=gun_a, pos=gun_set_pos, rgba=rgba)
            self.ammo_g = self.gun_set_b.add("geom", type="mesh",
                mesh=ammo_a, pos=gun_set_pos, rgba=rgba)
            self.cams_g = self.gun_set_b.add("geom", type="mesh",
                mesh=cams_a, pos=gun_set_pos, rgba=rgba)
            self.gun_set_j = self.gun_set_b.add("joint",
                pos=gun_set_pos, axis=(0, 1, 0), limited="true",
                range=(-1, 0.4))
            self.model.actuator.add('motor', gear=(100, 0, 0),
                                    joint=self.gun_set_j)

        # Attach legs to equidistant sites on the circumference
        for i in range(N_LEGS):
            theta = i * 2*np.pi / N_LEGS
            hip_pos = [0.2*np.cos(theta), 0.2*np.sin(theta), 0]
            hip_site = self.model.worldbody.add('site', pos=hip_pos,
                                           euler=[0, 0, theta])
            leg = Leg()
            hip_site.attach(leg.model)
        return self.model

    def _build(self):
        self.model = self.make_creature()

    def _build_observables(self):
        return CreatureObservables(self)

    @property
    def mjcf_model(self):
        return self.model

    @property
    def actuators(self):
        return tuple(self.model.find_all('actuator'))


# Add simple observable features for joint angles and velocities.
class CreatureObservables(composer.Observables):

    @composer.observable
    def base_position(self):
        # base = self._entity.mjcf_model.find('geom', 'base_g')
        return observable.MJCFFeature('xpos', self._entity.base_g)

    @composer.observable
    def base_orientation(self):
        return observable.MJCFFeature('xmat',
                                      self._entity.root_body)[8]

    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @composer.observable
    def base_velocities(self):
        # 3 orientational vels followed by 3 positional vels
        base_b = self._entity.base_b
        return observable.MJCFFeature('cvel', base_b)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class Stand(composer.Task):
    """Returns to go down 1st, as doing nothing is pretty
    rewarding in this task."""

    def __init__(self, creature):
        self._creature = creature
        self._arena = floors.Floor()
        self._arena._ground_geom.pos = (0, 0, -0.5)
        self._arena._ground_geom.friction = (0.005, 0.005, 0.0001)

        self._arena.add_free_entity(self._creature)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

        # Configure and enable observables
        self._creature.observables.base_position.enabled = True
        self._creature.observables.base_orientation.enabled = True
        self._creature.observables.joint_positions.enabled = True
        self._creature.observables.joint_velocities.enabled = True
        self._creature.observables.base_velocities.enabled = True
        # self._creature.observables.touch_sensors.enabled = False
        self._task_observables = {}

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = N_SUBSTEPS * 0.002

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        pass

    def initialize_episode(self, physics, random_state):
        if RANDOM_STARTS:
            self._creature.set_pose(physics,
                position=np.random.normal(scale=0.1, size=3),
                quaternion=np.random.normal(loc=(0, 0, 0, 1),
                scale=(1, 0.05, 0.05, 1), size=4))
        else:
            self._creature.set_pose(physics, position=(0, 0, 0))

    def get_reward(self, physics):
        z = physics.named.data.xpos['unnamed_model/base_b'][2]
        reward = -abs(z - PERFECT_Z) + 1
        # print(reward)
        return reward


class Walk(composer.Task):
    """Returns to go down 1st, as doing nothing is pretty
    rewarding in this task."""

    def __init__(self, creature):
        self._creature = creature
        self._arena = floors.Floor()
        self._arena._ground_geom.pos = (0, 0, -0.5)
        self._arena._ground_geom.friction = (0.005, 0.005, 0.0001)

        self._arena.add_free_entity(self._creature)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

        # Configure and enable observables
        self._creature.observables.base_position.enabled = True
        self._creature.observables.base_orientation.enabled = True
        self._creature.observables.joint_positions.enabled = True
        self._creature.observables.joint_velocities.enabled = True
        self._creature.observables.base_velocities.enabled = True
        # self._creature.observables.touch_sensors.enabled = False
        self._task_observables = {}

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = N_SUBSTEPS * 0.002

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        pass

    def initialize_episode(self, physics, random_state):
        if RANDOM_STARTS:
            self._creature.set_pose(physics,
                position=np.random.normal(scale=0.1, size=3),
                quaternion=np.random.normal(loc=(0, 0, 0, 1),
                scale=(1, 0.05, 0.05, 1), size=4))
        else:
            self._creature.set_pose(physics, position=(0, 0, 0))

    def get_reward(self, physics):
        # Getting reward for x
        reward = physics.named.data.xpos['unnamed_model/base_b'][0]
        # print(reward)
        return reward


creature = Creature()
if TASK == 'stand':
    task = Stand(creature)
elif TASK == 'walk':
    task = Walk(creature)
env = composer.Environment(task,
                           random_state=np.random.RandomState(42),
                           time_limit=TIME_LIMIT)
env.reset()

if __name__ == '__main__':
    viewer.launch(env)
