"""
BD bipedal robot environment — LIPM footstep planning + PPO.

Inherits directly from LeggedRobot (peer of HumanoidController). All stepping /
XCoM / rewards logic is self-contained here — no dependency on
HumanoidController.

BD body name map (11 rigid bodies in URDF order):
  idx  0: base_link
  idx  1: L0_Link          ← LEFT  hip-yaw body
  idx  2: L1_Link
  idx  3: L2_Link
  idx  4: L3_Link
  idx  5: L4_Link_ankle    ← LEFT  foot body
  idx  6: R0_Link          ← RIGHT hip-yaw body
  idx  7: R1_Link
  idx  8: R2_Link
  idx  9: R3_Link
  idx 10: R4_Link_ankle    ← RIGHT foot body

feet_ids ordering (driven by cfg.asset.end_effectors):
  feet_ids[0] = R4_Link_ankle  (RIGHT foot, index 0)
  feet_ids[1] = L4_Link_ankle  (LEFT  foot, index 1)
"""

import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse

from gym.envs.base.legged_robot import LeggedRobot
from gym.envs.lipm import LIPMStepPlanner
from gym.utils import XCoMKeyboardInterface
from gym.utils.math import wrap_to_pi

from .bd_controller_config import BDControllerCfg
from .bd_rewards import BDRewards
from .bd_terrain import BDTerrainAdapter
from .bd_visualization import BDVisualization


class BDController(BDRewards, BDVisualization, BDTerrainAdapter,
                   LIPMStepPlanner, LeggedRobot):
    cfg: BDControllerCfg

    # ------------------------------------------------------------------ #
    # BD-specific body names and constants                                 #
    # ------------------------------------------------------------------ #
    _RIGHT_HIP_BODY = 'R0_Link'         # hip-yaw body, right leg
    _LEFT_HIP_BODY  = 'L0_Link'         # hip-yaw body, left leg
    _RIGHT_FOOT_BODY = 'R4_Link_ankle'  # end-effector, right leg
    _LEFT_FOOT_BODY  = 'L4_Link_ankle'  # end-effector, left leg

    # Offset from ankle-joint frame to ground contact point [m].
    # BD ankle CoM is ~29.6 mm below joint; contact ~35 mm below joint.
    _FOOT_HEIGHT_OFFSET = -0.035

    # Minimum allowed distance between planned step commands [m].
    # Matches BD hip width (0.107 m) with small margin.
    _FOOT_COLLISION_THRESHOLD = 0.15

    # Initial stance half-width [m] — half of hip-to-hip distance for BD.
    _INIT_STANCE_HALF_WIDTH = 0.054

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------ #
    # Keyboard interface                                                   #
    # ------------------------------------------------------------------ #
    def _setup_keyboard_interface(self):
        self.keyboard_interface = XCoMKeyboardInterface(self)

    # ------------------------------------------------------------------ #
    # Buffers                                                              #
    # ------------------------------------------------------------------ #
    def _init_buffers(self):
        super()._init_buffers()

        # ---- Robot states ----
        self.base_height = self.root_states[:, 2:3]
        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx[self._RIGHT_HIP_BODY], :3]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx[self._LEFT_HIP_BODY], :3]
        self.foot_states = torch.zeros(self.num_envs, len(self.feet_ids), 7,
                                       dtype=torch.float, device=self.device,
                                       requires_grad=False)
        self.foot_states_right = torch.zeros(self.num_envs, 4, dtype=torch.float,
                                             device=self.device, requires_grad=False)
        self.foot_states_left = torch.zeros(self.num_envs, 4, dtype=torch.float,
                                            device=self.device, requires_grad=False)
        self.foot_heading = torch.zeros(self.num_envs, len(self.feet_ids),
                                        dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.foot_projected_gravity = torch.stack(
            (self.gravity_vec, self.gravity_vec), dim=1)
        self.foot_contact = torch.zeros(self.num_envs, len(self.feet_ids),
                                        dtype=torch.bool, device=self.device,
                                        requires_grad=False)
        self.ankle_vel_history = torch.zeros(self.num_envs, len(self.feet_ids), 2 * 3,
                                             dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.base_heading = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                        device=self.device, requires_grad=False)
        self.base_lin_vel_world = torch.zeros(self.num_envs, 3, dtype=torch.float,
                                              device=self.device, requires_grad=False)

        # ---- Step commands ----
        self.step_commands = torch.zeros(self.num_envs, len(self.feet_ids), 3,
                                         dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.step_commands_right = torch.zeros(self.num_envs, 4, dtype=torch.float,
                                               device=self.device, requires_grad=False)
        self.step_commands_left = torch.zeros(self.num_envs, 4, dtype=torch.float,
                                              device=self.device, requires_grad=False)
        self.foot_on_motion = torch.zeros(self.num_envs, len(self.feet_ids),
                                          dtype=torch.bool, device=self.device,
                                          requires_grad=False)
        self.step_period = torch.zeros(self.num_envs, 1, dtype=torch.long,
                                       device=self.device, requires_grad=False)
        self.full_step_period = torch.zeros(self.num_envs, 1, dtype=torch.long,
                                            device=self.device, requires_grad=False)
        self.ref_foot_trajectories = torch.zeros(self.num_envs, len(self.feet_ids), 3,
                                                 dtype=torch.float, device=self.device,
                                                 requires_grad=False)

        # ---- Step states ----
        self.current_step = torch.zeros(self.num_envs, len(self.feet_ids), 3,
                                        dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.prev_step_commands = torch.zeros(self.num_envs, len(self.feet_ids), 3,
                                              dtype=torch.float, device=self.device,
                                              requires_grad=False)
        self.step_location_offset = torch.zeros(self.num_envs, len(self.feet_ids),
                                                dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.step_heading_offset = torch.zeros(self.num_envs, len(self.feet_ids),
                                               dtype=torch.float, device=self.device,
                                               requires_grad=False)
        self.succeed_step_radius = torch.tensor(
            self.cfg.commands.succeed_step_radius,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.succeed_step_angle = torch.tensor(
            np.deg2rad(self.cfg.commands.succeed_step_angle),
            dtype=torch.float, device=self.device, requires_grad=False)
        self.semi_succeed_step = torch.zeros(self.num_envs, len(self.feet_ids),
                                             dtype=torch.bool, device=self.device,
                                             requires_grad=False)
        self.succeed_step = torch.zeros(self.num_envs, len(self.feet_ids),
                                        dtype=torch.bool, device=self.device,
                                        requires_grad=False)
        self.already_succeed_step = torch.zeros(self.num_envs, dtype=torch.bool,
                                                device=self.device, requires_grad=False)
        self.had_wrong_contact = torch.zeros(self.num_envs, len(self.feet_ids),
                                             dtype=torch.bool, device=self.device,
                                             requires_grad=False)
        self.step_stance = torch.zeros(self.num_envs, 1, dtype=torch.long,
                                       device=self.device, requires_grad=False)

        # ---- Others ----
        self.update_count = torch.zeros(self.num_envs, dtype=torch.long,
                                        device=self.device, requires_grad=False)
        self.update_commands_ids = torch.zeros(self.num_envs, dtype=torch.bool,
                                               device=self.device, requires_grad=False)
        self.phase_count = torch.zeros(self.num_envs, dtype=torch.long,
                                       device=self.device, requires_grad=False)
        self.update_phase_ids = torch.zeros(self.num_envs, dtype=torch.bool,
                                            device=self.device, requires_grad=False)
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device, requires_grad=False)

        # ---- LIPM buffers (CoM, ICP, w, LIPM_CoM, support_foot_pos, raibert, step_length/width) ----
        self._init_lipm_buffers()

        # ---- Observation variables ----
        self.phase_sin = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                     device=self.device, requires_grad=False)
        self.phase_cos = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                     device=self.device, requires_grad=False)
        self.contact_schedule = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                            device=self.device, requires_grad=False)

    # ------------------------------------------------------------------ #
    # Reset / resample                                                     #
    # ------------------------------------------------------------------ #
    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)

        self.step_period[env_ids] = torch.randint(
            low=self.command_ranges["sample_period"][0],
            high=self.command_ranges["sample_period"][1],
            size=(len(env_ids), 1), device=self.device)
        self.full_step_period = 2 * self.step_period

        self.step_stance[env_ids] = torch.clone(self.step_period[env_ids])

        from isaacgym.torch_utils import torch_rand_float
        self.dstep_width[env_ids] = torch_rand_float(
            self.command_ranges["dstep_width"][0],
            self.command_ranges["dstep_width"][1],
            (len(env_ids), 1), self.device)

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)

        # Robot states
        self.foot_states[env_ids] = self._calculate_foot_states(
            self.rigid_body_state[:, self.feet_ids, :7])[env_ids]
        self.foot_projected_gravity[env_ids, 0] = self.gravity_vec[env_ids]
        self.foot_projected_gravity[env_ids, 1] = self.gravity_vec[env_ids]

        # Initial step commands: feet_ids[0]=RIGHT, feet_ids[1]=LEFT
        half_w = self._INIT_STANCE_HALF_WIDTH
        self.step_commands[env_ids, 0] = (
            self.env_origins[env_ids]
            + torch.tensor([0., -half_w, 0.], device=self.device))
        self.step_commands[env_ids, 1] = (
            self.env_origins[env_ids]
            + torch.tensor([0.,  half_w, 0.], device=self.device))
        self.foot_on_motion[env_ids, 0] = False
        self.foot_on_motion[env_ids, 1] = True  # left foot starts as swing

        # Step states
        self.current_step[env_ids] = torch.clone(self.step_commands[env_ids])
        self.prev_step_commands[env_ids] = torch.clone(self.step_commands[env_ids])
        self.semi_succeed_step[env_ids] = False
        self.succeed_step[env_ids] = False
        self.already_succeed_step[env_ids] = False
        self.had_wrong_contact[env_ids] = False

        # Others
        self.update_count[env_ids] = 0
        self.update_commands_ids[env_ids] = False
        self.phase_count[env_ids] = 0
        self.update_phase_ids[env_ids] = False
        self.phase[env_ids] = 0

        # LIPM state (ICP, w, raibert_heuristic, dstep_length/width)
        self._reset_lipm_buffers(env_ids)

    # ------------------------------------------------------------------ #
    # Per-step callback                                                    #
    # ------------------------------------------------------------------ #
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        self._update_robot_states()
        self._calculate_CoM()
        self._calculate_raibert_heuristic()
        self._calculate_ICP()
        self._measure_success_rate()
        self._update_commands()

    def _update_robot_states(self):
        """Update robot state variables."""
        self.base_height[:] = self.root_states[:, 2:3]
        forward = quat_apply(self.base_quat, self.forward_vec)
        self.base_heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)

        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx[self._RIGHT_HIP_BODY], :3]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx[self._LEFT_HIP_BODY], :3]

        self.foot_states = self._calculate_foot_states(
            self.rigid_body_state[:, self.feet_ids, :7])

        right_foot_forward = quat_apply(self.foot_states[:, 0, 3:7], self.forward_vec)
        left_foot_forward = quat_apply(self.foot_states[:, 1, 3:7], self.forward_vec)
        self.foot_heading[:, 0] = wrap_to_pi(
            torch.atan2(right_foot_forward[:, 1], right_foot_forward[:, 0]))
        self.foot_heading[:, 1] = wrap_to_pi(
            torch.atan2(left_foot_forward[:, 1], left_foot_forward[:, 0]))

        self.foot_projected_gravity[:, 0] = quat_rotate_inverse(
            self.foot_states[:, 0, 3:7], self.gravity_vec)
        self.foot_projected_gravity[:, 1] = quat_rotate_inverse(
            self.foot_states[:, 1, 3:7], self.gravity_vec)

        self.update_count += 1
        self.phase_count += 1
        self.phase += 1 / self.full_step_period

        self.foot_contact = torch.gt(self.contact_forces[:, self.feet_ids, 2], 0)
        self.contact_schedule = self.smooth_sqr_wave(self.phase)

        # Update current step from contact
        current_step_masked = self.current_step[self.foot_contact]
        current_step_masked[:, :2] = self.foot_states[self.foot_contact][:, :2]
        current_step_masked[:, 2] = self.foot_heading[self.foot_contact]
        self.current_step[self.foot_contact] = current_step_masked

        naxis = 3
        self.ankle_vel_history[:, 0, naxis:] = self.ankle_vel_history[:, 0, :naxis]
        self.ankle_vel_history[:, 0, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx[self._RIGHT_FOOT_BODY], 7:10]
        self.ankle_vel_history[:, 1, naxis:] = self.ankle_vel_history[:, 1, :naxis]
        self.ankle_vel_history[:, 1, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx[self._LEFT_FOOT_BODY], 7:10]

    def _calculate_foot_states(self, foot_states):
        """Adjust foot position by BD-specific ankle-to-contact offset."""
        foot_height_vec = torch.tensor(
            [0., 0., self._FOOT_HEIGHT_OFFSET]
        ).repeat(self.num_envs, 1).to(self.device)
        rfoot_h = quat_apply(foot_states[:, 0, 3:7], foot_height_vec)
        lfoot_h = quat_apply(foot_states[:, 1, 3:7], foot_height_vec)
        foot_states[:, 0, :3] += rfoot_h
        foot_states[:, 1, :3] += lfoot_h
        return foot_states

    def _measure_success_rate(self):
        """Measure success rate of step commands."""
        zeros = torch.zeros((self.num_envs, len(self.feet_ids), 1), device=self.device)

        self.step_location_offset = torch.norm(
            self.foot_states[:, :, :3]
            - torch.cat((self.step_commands[:, :, :2], zeros), dim=2), dim=2)
        self.step_heading_offset = torch.abs(
            wrap_to_pi(self.foot_heading - self.step_commands[:, :, 2]))
        self.semi_succeed_step = (
            (self.step_location_offset < self.succeed_step_radius)
            & (self.step_heading_offset < self.succeed_step_angle))

        self.prev_step_location_offset = torch.norm(
            self.foot_states[:, :, :3]
            - torch.cat((self.prev_step_commands[:, :, :2], zeros), dim=2), dim=2)
        self.prev_step_heading_offset = torch.abs(
            wrap_to_pi(self.foot_heading - self.prev_step_commands[:, :, 2]))
        self.prev_semi_succeed_step = (
            (self.prev_step_location_offset < self.succeed_step_radius)
            & (self.prev_step_heading_offset < self.succeed_step_angle))

        self.had_wrong_contact |= (
            self.foot_contact * ~self.semi_succeed_step * ~self.prev_semi_succeed_step)

        self.succeed_step = self.semi_succeed_step & ~self.had_wrong_contact
        self.succeed_step_ids = (self.succeed_step.sum(dim=1) == 2)
        self.already_succeed_step[self.succeed_step_ids] = True

    def _update_commands(self):
        """Update step commands (BD-specific collision threshold)."""
        self.update_phase_ids = (
            self.phase_count >= self.full_step_period.squeeze(1))
        self.phase_count[self.update_phase_ids] = 0
        self.phase[self.update_phase_ids] = 0

        self.update_commands_ids = (
            self.update_count >= self.step_period.squeeze(1))
        self.already_succeed_step[self.update_commands_ids] = False
        self.had_wrong_contact[self.update_commands_ids] = False
        self.update_count[self.update_commands_ids] = 0
        self.step_stance[self.update_commands_ids] = torch.clone(
            self.step_period[self.update_commands_ids])

        self.foot_on_motion[self.update_commands_ids] = (
            ~self.foot_on_motion[self.update_commands_ids])

        update_step_commands_mask = self.step_commands[self.update_commands_ids]
        self.prev_step_commands[self.update_commands_ids] = torch.clone(
            self.step_commands[self.update_commands_ids])

        update_step_commands_mask[
            self.foot_on_motion[self.update_commands_ids]
        ] = self._generate_step_command_by_3DLIPM_XCoM(self.update_commands_ids)
        self._update_LIPM_CoM(self.update_commands_ids)

        foot_collision_ids = (
            update_step_commands_mask[:, 0, :2]
            - update_step_commands_mask[:, 1, :2]
        ).norm(dim=1) < self._FOOT_COLLISION_THRESHOLD
        update_step_commands_mask[foot_collision_ids, :, :2] = (
            self._adjust_foot_collision(
                update_step_commands_mask[foot_collision_ids, :, :2],
                self.foot_on_motion[self.update_commands_ids][foot_collision_ids]))

        if self.cfg.terrain.measure_heights:
            update_step_commands_mask[
                self.foot_on_motion[self.update_commands_ids]
            ] = self._adjust_step_command_in_rough_terrain(
                self.update_commands_ids, update_step_commands_mask)

        self.step_commands[self.update_commands_ids] = update_step_commands_mask

    # ------------------------------------------------------------------ #
    # Foot collision correction                                            #
    # ------------------------------------------------------------------ #
    def _adjust_foot_collision(self, collision_step_commands, collision_foot_on_motion):
        """Place swing foot at _FOOT_COLLISION_THRESHOLD distance from support."""
        collision_distance = (
            collision_step_commands[:, 0] - collision_step_commands[:, 1]
        ).norm(dim=1, keepdim=True)
        adjust_step_commands = collision_step_commands.clone()
        adjust_step_commands[collision_foot_on_motion] = (
            collision_step_commands[~collision_foot_on_motion]
            + self._FOOT_COLLISION_THRESHOLD
            * (collision_step_commands[collision_foot_on_motion]
               - collision_step_commands[~collision_foot_on_motion])
            / collision_distance)
        return adjust_step_commands

    def _calculate_foot_ref_trajectory(self, prev_step_commands, step_commands):
        """Paraboloid reference foot trajectory between prev and next step."""
        center = (step_commands[:, :, :2] + prev_step_commands[:, :, :2]) / 2
        radius = (step_commands[:, :, :2] - prev_step_commands[:, :, :2]).norm(dim=2) / 2
        apex_height = self.cfg.commands.apex_height_percentage * radius
        self.ref_foot_trajectories[:, :, :2] = (
            prev_step_commands[:, :, :2]
            + (step_commands[:, :, :2] - prev_step_commands[:, :, :2])
            * self.phase.unsqueeze(2))
        a = radius / apex_height.sqrt()
        self.ref_foot_trajectories[:, :, 2] = (
            -torch.sum(torch.square(self.ref_foot_trajectories[:, :, :2] - center),
                       dim=2) / a.square() + apex_height)
        self.ref_foot_trajectories[:, :, 2][
            torch.isnan(self.ref_foot_trajectories[:, :, 2])] = 0

    # ------------------------------------------------------------------ #
    # Curriculum stubs                                                     #
    # ------------------------------------------------------------------ #
    def _update_command_curriculum(self, env_ids):
        pass

    def _update_reward_curriculum(self, env_ids):
        pass

    # ------------------------------------------------------------------ #
    # Observation variables                                                #
    # ------------------------------------------------------------------ #
    def _set_obs_variables(self):
        zeros = torch.zeros((self.num_envs, 1), device=self.device)
        self.foot_states_right[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 0, :3] - self.base_pos)
        self.foot_states_left[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 1, :3] - self.base_pos)
        self.foot_states_right[:, 3] = wrap_to_pi(
            self.foot_heading[:, 0] - self.base_heading.squeeze(1))
        self.foot_states_left[:, 3] = wrap_to_pi(
            self.foot_heading[:, 1] - self.base_heading.squeeze(1))

        self.step_commands_right[:, :3] = quat_rotate_inverse(
            self.base_quat,
            torch.cat((self.step_commands[:, 0, :2], zeros), dim=1) - self.base_pos)
        self.step_commands_left[:, :3] = quat_rotate_inverse(
            self.base_quat,
            torch.cat((self.step_commands[:, 1, :2], zeros), dim=1) - self.base_pos)
        self.step_commands_right[:, 3] = wrap_to_pi(
            self.step_commands[:, 0, 2] - self.base_heading.squeeze(1))
        self.step_commands_left[:, 3] = wrap_to_pi(
            self.step_commands[:, 1, 2] - self.base_heading.squeeze(1))

        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.base_lin_vel_world = self.root_states[:, 7:10].clone()

    # ------------------------------------------------------------------ #
    # Termination                                                          #
    # ------------------------------------------------------------------ #
    def check_termination(self):
        """Termination with BD-appropriate thresholds.

        BD is much lighter than MIT Humanoid → same joint torques produce higher
        angular acceleration → ang_vel threshold raised from 5 → 8 rad/s.
        Base-height check omitted: nominal standing height (0.30 m) is close to
        the 0.3 m floor that would fire during normal walking.
        """
        term_contact = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
        self.terminated = torch.any((term_contact > 1.), dim=1)
        self.terminated |= torch.any(
            torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1)
        self.terminated |= torch.any(
            torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 8., dim=1)
        self.terminated |= torch.any(
            torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)
        self.terminated |= torch.any(
            torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)

        self.timed_out = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.terminated | self.timed_out

    def post_physics_step(self):
        super().post_physics_step()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def smooth_sqr_wave(phase):
        p = 2. * torch.pi * phase
        eps = 0.2
        return torch.sin(p) / torch.sqrt(torch.sin(p) ** 2. + eps ** 2.)
