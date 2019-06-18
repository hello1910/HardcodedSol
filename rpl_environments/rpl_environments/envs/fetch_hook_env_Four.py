from gym.envs.robotics import rotations, fetch_env
from gym import utils, spaces
import numpy as np
import os
import mujoco_py
from gym import error
from .hook_controller import get_hook_control, block_is_grasped, block_inside_grippers, grippers_are_closed, grippers_are_open, pick_at_position

import pdb

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DEBUG=True

def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()
    




def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
def ctrl_(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta



class FetchHookEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, xml_file=None):
        object_x=1.7
        object_y=1
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [object_x,object_y, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.35, 0.35, 0.4, 1., 0., 0., 0.],
        }

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, 'assets', 'hookExpFour.xml')

        #self._goal_pos = np.array([1.4, 1, 0.42])
        self._goal_pos=np.array([2.1,0.48, 0.42])
        self.target_score=0
        self.count=0
        self._object_xpos = np.array([object_x,object_y])

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')
        
        utils.EzPickle.__init__(self)

    def get_move_action(self,observation, target_position, atol=0.0012, gain=10, close_gripper=False, move=False):
    # Get the currents
        current_position = observation['observation'][:3]
        if move:
           #target_save=target_position object: 1.5, 1
           target_position=[0.8,1,0.4] #first, lower number means back towards robot
           #second, higher number means to the right, from your view
            #print((current_position[0]-target_position[0])**2 + (current_position[1]-target_position[1])**2 + (current_position[2]-target_position[2])**2)
           if (current_position[0]-target_position[0])**2 + (current_position[1]-target_position[1])**2 + (current_position[2]-target_position[2])**2 < atol:
                if self.target_score==0:
                    self.target_score=1
           if self.target_score==1:
                target_position= [1.6,1,0.4]
                self.count=self.count+1
                gain=6.8
                if (self.count==60):
                    self.target_score=2
                    self.count=0
                if (current_position[0]-target_position[0])**2 + (current_position[1]-target_position[1])**2 + (current_position[2]-target_position[2])**2 < atol:
                    #pass
                    if self.target_score==1:
                        self.target_score=2
                        print("ARRIVED")
           if self.target_score==2:
                target_position=[2,1.2,0.42]
                self.count+=1
                if (self.count==50):
                    self.target_score=3
                    self.count=0
           if self.target_score==3:
               target_position=[1.4,1,0.42]
               self.count=self.count+1
               if (self.count==70):
                   self.target_score=4
           if self.target_score==4:
               target_position=[1.4,0.7,0.42]
               gain=5.5

        action = gain * np.subtract(target_position, current_position) 
        if close_gripper:
        	    gripper_action=-1
        else:
             gripper_action=0
        action=np.hstack((action,gripper_action))
        return action
    
    
    
    def get_hook_control(self,obs, atol=1e-2):
        """
        Returns
        -------
        action : [float] * 4
        """

        gripper_position = obs['observation'][:3]
        block_position = obs['observation'][3:6]
        hook_position = obs['observation'][25:28]
        place_position = obs['desired_goal']
    
        # Done
        if abs(block_position[0] - place_position[0]) + abs(block_position[1] - place_position[1]) <= 1e-2:
            if DEBUG:
                print("DONE")
            return np.array([0., 0., 0., -1.])
    
        # Grasp and lift the hook
        if not block_is_grasped(obs, gripper_position, hook_position, relative_grasp_position=(0., 0., -0.05), atol=atol):
            if DEBUG:
                print("Grasping and lifting the hook")
            hook_target = hook_position.copy()
            hook_target[2] = 0.5
            return pick_at_position(obs, hook_position, hook_target, relative_grasp_position=(0., 0., -0.05))
    
        # Align the hook to sweep
        hook_target = np.array([block_position[0] - 0.5, block_position[1] - 0.05, block_position[2]])
    
        #if hook_position[0] >= hook_target[0] + 0.1 or hook_position[1] + 0.1 <= hook_target[1]:
        if DEBUG:
            #print("Aligning to sweep", hook_position[0] + atol, hook_target[0], hook_position[1] + atol, hook_target[1])
            print(self.target_score)
            return self.get_move_action(obs, hook_target, close_gripper=True,move=True)
    
        if DEBUG:
            print("Sweeping back")
    
        #direction = np.subtract(place_position, block_position)
        #direction = direction[:2] / np.linalg.norm(direction[:2])
    
        #return np.array([0.4 * direction[0], 0.4 * direction[1], 0., -1.])




    def render(self, mode="human", *args, **kwargs):
        # See https://github.com/openai/gym/issues/1081
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer('rgb_array').render()
            width, height = 3350, 1800
            data = self._get_viewer('rgb_array').read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer('human').render()

        return super(FetchHookEnv, self).render(*args, **kwargs)
        
    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        if self.set_quat:
            #rot_ctrl=[0.866, 0.353, 0.353, 0]
        	    rot_ctrl=[1, 0, 1., 0.] 
        else:
            rot_ctrl = [1, 0., 1, 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        ctrl_(self.sim, action)
        mocap_(self.sim, action)

    def _sample_goal(self):
        goal_pos = self._goal_pos.copy()
        goal_pos[:2] += self.np_random.uniform(-0.05, 0.05)
        return goal_pos

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = -24.

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        num_1=self._object_xpos[0]
        num_2=self._object_xpos[1]
        while True:
            object_xpos_x = num_1 + self.np_random.uniform(-0.05, 0.10)
            object_xpos_y = num_2 + self.np_random.uniform(-0.05, 0.05)
            if (object_xpos_x - self._goal_pos[0])**2 + (object_xpos_y - self._goal_pos[1])**2 >= 0.01:
                break
        self._object_xpos = np.array([object_xpos_x, object_xpos_y])

        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = self._object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
      
        return obs

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)


class FetchHookEnvTwoEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.85, 0.8, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.60, 0.35, 0.4, 1., 0., 0., 0.],
        }

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, 'assets', 'hookExpTwo.xml')

        self._goal_pos = np.array([1.3, 0.8, 0.42])
        self._object_xpos = np.array([1.85, 0.8])

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')
        
        utils.EzPickle.__init__(self)
    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        
        hook_pos = self.sim.data.get_site_xpos('object1')
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object1'))
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('object1') * dt
        hook_velr = self.sim.data.get_site_xvelr('object1') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])

        return obs
        
     
class FetchHookEnvOneEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.8, 0.75, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.35, 0.35, 0.4, 1., 0., 0., 0.],
        }

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, 'assets', 'hookExpOne.xml')

        self._goal_pos = np.array([1.50, 1.1, 0.42])
        self._object_xpos = np.array([1.8, 1.1])

        #self._goal_pos = np.array([1.5, 0.8, 0.42])
        #self._object_xpos = np.array([1.7, 1])

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')
        
        utils.EzPickle.__init__(self)
        
    def step(self, residual_action):
        #residual_action = 2. * residual_action

        #action = np.add(residual_action, get_hook_control(self._last_observation))
        action,bool=get_hook_control(self._last_observation)
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)
        
        self._last_observation = observation
        
        return observation, reward, done, debug_info

    def compute_reward(self, *args, **kwargs):
        return FetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation

        return observation
        
class FetchHookEnvThreeEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        object_x=2.05
        object_y=0.8
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [object_x, object_y, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.60, 0.35, 0.4, 1., 0., 0., 0.],
        }

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, 'assets', 'hookExpThree.xml')

        self._goal_pos = np.array([1.3, 0.83, 0.42])
        self._object_xpos = np.array([object_x, 0.75])

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')
        
        utils.EzPickle.__init__(self)
        
class FetchHookEnvSixEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        object_x=2.05
        object_y=0.8
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [object_x,object_y, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.50, 0.45, 0.4, 1., 0., 0., 0.],
            'hooktwo:joint':[1.9,0.8,0.4,1.,0.,0.,0.],
            
        }

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, 'assets', 'hookExpSix.xml')

        self._goal_pos = np.array([1.3, 0.83, 0.42])

        self._object_xpos = np.array([object_x,object_y])

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')
        
        utils.EzPickle.__init__(self)
        
    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        
        hook_pos = self.sim.data.get_site_xpos('hooktwo')
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hooktwo'))
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hooktwo') * dt
        hook_velr = self.sim.data.get_site_xvelr('hooktwo') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])

        return obs

class ResidualFetchHookEnv(FetchHookEnv):

    def step(self, residual_action):
        #residual_action = 2. * residual_action

        #action = np.add(residual_action, get_hook_control(self._last_observation))
        
        action=self.get_hook_control(self._last_observation)
        
        action = np.clip(action, -1, 1)
        if bool:
            self.set_quat=True
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)
        
        self._last_observation = observation
        
        return observation, reward, done, debug_info

    def compute_reward(self, *args, **kwargs):
        return FetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation

        return observation

class FetchHookEnvFourEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        object_x=1.5
        object_y=1
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [object_x,object_y, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.50, 0.45, 0.4, 1., 0., 0., 0.],
        }

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, 'assets', 'hookExpFour.xml')

        #self._goal_pos = np.array([1.65, 0.75, 0.42])
        self._goal_pos=np.array([2,0.45, 0.42])
        
        self._object_xpos = np.array([object_x,object_y])

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')
        
        utils.EzPickle.__init__(self)
        
class FetchHookEnvFiveEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        object_x=1.5
        object_y=1
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [object_x,object_y, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.50, 0.45, 0.4, 1., 0., 0., 0.],
        }

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, 'assets', 'hookExpFive.xml')

        #self._goal_pos = np.array([1.65, 0.75, 0.42])
        self._goal_pos=np.array([2,0.45, 0.42])
        
        self._object_xpos = np.array([object_x,object_y])

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')
        
        utils.EzPickle.__init__(self)

class NoisyFetchHookEnv(FetchHookEnv):

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]#object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

    def compute_reward(self, *args, **kwargs):
        return FetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation

        return observation

class TwoFrameResidualHookNoisyEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        super(TwoFrameResidualHookNoisyEnv, self).__init__()
        self.observation_space.spaces['observation'] = spaces.Box(low=np.hstack((self.observation_space.spaces['observation'].low, self.observation_space.spaces['observation'].low)), 
            high=np.hstack((self.observation_space.spaces['observation'].high, self.observation_space.spaces['observation'].high)),dtype=np.float32)
    
    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]#object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)
        
        obs_out = observation.copy()
        obs_out['observation'] = np.hstack((self._last_observation['observation'], observation['observation'])) 
        self._last_observation = observation
        
        return obs_out, reward, done, debug_info

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation.copy()
        observation['observation'] = np.hstack((self._last_observation['observation'], observation['observation']))
        return observation

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

class TwoFrameHookNoisyEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        super(TwoFrameHookNoisyEnv, self).__init__()
        self.observation_space.spaces['observation'] = spaces.Box(low=np.hstack((self.observation_space.spaces['observation'].low, self.observation_space.spaces['observation'].low)), 
            high=np.hstack((self.observation_space.spaces['observation'].high, self.observation_space.spaces['observation'].high)),dtype=np.float32)
    
    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]#object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def step(self, action):
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)
        
        obs_out = observation.copy()
        obs_out['observation'] = np.hstack((self._last_observation['observation'], observation['observation'])) 
        self._last_observation = observation
        
        return obs_out, reward, done, debug_info

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation.copy()
        observation['observation'] = np.hstack((self._last_observation['observation'], observation['observation']))
        return observation

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

class NoisyResidualFetchHookEnv(FetchHookEnv):

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)
        
        self._last_observation = observation
        
        return observation, reward, done, debug_info

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]#object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

    def compute_reward(self, *args, **kwargs):
        return FetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation

        return observation
