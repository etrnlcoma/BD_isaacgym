"""
BD bipedal robot environment — LIPM footstep planning + PPO.

Subclasses HumanoidController and overrides all hardcoded body-name
references (hip joints, foot bodies) to match BD.urdf naming.

BD body name map (11 rigid bodies in URDF order):
  idx  0: base_link
  idx  1: L0_Link          ← LEFT  hip-yaw body  (was 'right_hip_yaw' in MIT)
  idx  2: L1_Link
  idx  3: L2_Link
  idx  4: L3_Link
  idx  5: L4_Link_ankle    ← LEFT  foot body     (was 'right_foot' in MIT)
  idx  6: R0_Link          ← RIGHT hip-yaw body  (was 'left_hip_yaw' in MIT)
  idx  7: R1_Link
  idx  8: R2_Link
  idx  9: R3_Link
  idx 10: R4_Link_ankle    ← RIGHT foot body     (was 'left_foot' in MIT)

feet_ids ordering (driven by cfg.asset.end_effectors):
  feet_ids[0] = R4_Link_ankle  (RIGHT foot, index 0 ≡ original convention)
  feet_ids[1] = L4_Link_ankle  (LEFT  foot, index 1 ≡ original convention)

Because feet_ids[0] is RIGHT, all original HumanoidController logic is
preserved: step_commands[:,0] = right step, step_commands[:,1] = left step,
etc. Only the body-name lookups change.
"""

import torch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
from gym.envs.humanoid.humanoid_controller import HumanoidController
from gym.envs.humanoid.bd_controller_config import BDControllerCfg
from gym.utils.math import wrap_to_pi


class BDController(HumanoidController):
    cfg: BDControllerCfg

    # ------------------------------------------------------------------ #
    # Body names for BD.urdf                                               #
    # ------------------------------------------------------------------ #
    _RIGHT_HIP_BODY = 'R0_Link'        # hip-yaw body, right leg
    _LEFT_HIP_BODY  = 'L0_Link'        # hip-yaw body, left leg
    _RIGHT_FOOT_BODY = 'R4_Link_ankle' # end-effector, right leg
    _LEFT_FOOT_BODY  = 'L4_Link_ankle' # end-effector, left leg

    # Offset from ankle-joint frame to ground contact point [m].
    # BD ankle CoM is ~29.6 mm below joint; contact ~35 mm below joint.
    _FOOT_HEIGHT_OFFSET = -0.035       # m  (was -0.04 for MIT Humanoid)

    # Minimum allowed distance between planned step commands [m].
    # Matches BD hip width (0.107 m) with small margin.
    _FOOT_COLLISION_THRESHOLD = 0.10   # m  (was 0.2 for MIT Humanoid)

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------ #
    # Override _init_buffers to use BD body names                          #
    # ------------------------------------------------------------------ #
    def _init_buffers(self):
        # Skip HumanoidController._init_buffers() — it contains hardcoded
        # MIT Humanoid body names ('right_hip_yaw', etc.).
        # Call LeggedRobot._init_buffers() directly, then replicate all
        # HumanoidController buffer setup with BD-specific body names.
        from gym.envs.base.legged_robot import LeggedRobot
        LeggedRobot._init_buffers(self)

        # ---- Robot states (HumanoidController lines 42-54, BD names) ----
        self.base_height = self.root_states[:, 2:3]
        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx[self._RIGHT_HIP_BODY], :3]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx[self._LEFT_HIP_BODY], :3]
        self.CoM = torch.zeros(self.num_envs, 3, dtype=torch.float,
                               device=self.device, requires_grad=False)
        self.foot_states = torch.zeros(
            self.num_envs, len(self.feet_ids), 7,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_states_right = torch.zeros(
            self.num_envs, 4, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.foot_states_left = torch.zeros(
            self.num_envs, 4, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.foot_heading = torch.zeros(
            self.num_envs, len(self.feet_ids),
            dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_projected_gravity = torch.stack(
            (self.gravity_vec, self.gravity_vec), dim=1)
        self.foot_contact = torch.zeros(
            self.num_envs, len(self.feet_ids),
            dtype=torch.bool, device=self.device, requires_grad=False)
        self.ankle_vel_history = torch.zeros(
            self.num_envs, len(self.feet_ids), 2 * 3,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.base_heading = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.base_lin_vel_world = torch.zeros(
            self.num_envs, 3, dtype=torch.float,
            device=self.device, requires_grad=False)

        # ---- Step commands ----
        self.step_commands = torch.zeros(
            self.num_envs, len(self.feet_ids), 3,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.step_commands_right = torch.zeros(
            self.num_envs, 4, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.step_commands_left = torch.zeros(
            self.num_envs, 4, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.foot_on_motion = torch.zeros(
            self.num_envs, len(self.feet_ids),
            dtype=torch.bool, device=self.device, requires_grad=False)
        self.step_period = torch.zeros(
            self.num_envs, 1, dtype=torch.long,
            device=self.device, requires_grad=False)
        self.full_step_period = torch.zeros(
            self.num_envs, 1, dtype=torch.long,
            device=self.device, requires_grad=False)
        self.ref_foot_trajectories = torch.zeros(
            self.num_envs, len(self.feet_ids), 3,
            dtype=torch.float, device=self.device, requires_grad=False)

        # ---- Step states ----
        self.current_step = torch.zeros(
            self.num_envs, len(self.feet_ids), 3,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_step_commands = torch.zeros(
            self.num_envs, len(self.feet_ids), 3,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.step_location_offset = torch.zeros(
            self.num_envs, len(self.feet_ids),
            dtype=torch.float, device=self.device, requires_grad=False)
        self.step_heading_offset = torch.zeros(
            self.num_envs, len(self.feet_ids),
            dtype=torch.float, device=self.device, requires_grad=False)
        import numpy as np
        self.succeed_step_radius = torch.tensor(
            self.cfg.commands.succeed_step_radius,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.succeed_step_angle = torch.tensor(
            np.deg2rad(self.cfg.commands.succeed_step_angle),
            dtype=torch.float, device=self.device, requires_grad=False)
        self.semi_succeed_step = torch.zeros(
            self.num_envs, len(self.feet_ids),
            dtype=torch.bool, device=self.device, requires_grad=False)
        self.succeed_step = torch.zeros(
            self.num_envs, len(self.feet_ids),
            dtype=torch.bool, device=self.device, requires_grad=False)
        self.already_succeed_step = torch.zeros(
            self.num_envs, dtype=torch.bool,
            device=self.device, requires_grad=False)
        self.had_wrong_contact = torch.zeros(
            self.num_envs, len(self.feet_ids),
            dtype=torch.bool, device=self.device, requires_grad=False)
        self.step_stance = torch.zeros(
            self.num_envs, 1, dtype=torch.long,
            device=self.device, requires_grad=False)

        # ---- Others ----
        self.update_count = torch.zeros(
            self.num_envs, dtype=torch.long,
            device=self.device, requires_grad=False)
        self.update_commands_ids = torch.zeros(
            self.num_envs, dtype=torch.bool,
            device=self.device, requires_grad=False)
        self.phase_count = torch.zeros(
            self.num_envs, dtype=torch.long,
            device=self.device, requires_grad=False)
        self.update_phase_ids = torch.zeros(
            self.num_envs, dtype=torch.bool,
            device=self.device, requires_grad=False)
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.ICP = torch.zeros(
            self.num_envs, 3, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.raibert_heuristic = torch.zeros(
            self.num_envs, len(self.feet_ids), 3,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.w = torch.zeros(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.step_length = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.step_width = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.dstep_length = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.dstep_width = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.support_foot_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.prev_support_foot_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.LIPM_CoM = torch.zeros(
            self.num_envs, 3, dtype=torch.float,
            device=self.device, requires_grad=False)

        # ---- Observation variables ----
        self.phase_sin = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.phase_cos = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.contact_schedule = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)

    # ------------------------------------------------------------------ #
    # Override _update_robot_states to use BD body names                   #
    # ------------------------------------------------------------------ #
    def _update_robot_states(self):
        """Same as parent but references BD body names."""
        self.base_height[:] = self.root_states[:, 2:3]
        forward = quat_apply(self.base_quat, self.forward_vec)
        self.base_heading = torch.atan2(
            forward[:, 1], forward[:, 0]).unsqueeze(1)

        # BD body names for hip positions
        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx[self._RIGHT_HIP_BODY], :3]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx[self._LEFT_HIP_BODY], :3]

        self.foot_states = self._calculate_foot_states(
            self.rigid_body_state[:, self.feet_ids, :7])

        right_foot_forward = quat_apply(
            self.foot_states[:, 0, 3:7], self.forward_vec)
        left_foot_forward = quat_apply(
            self.foot_states[:, 1, 3:7], self.forward_vec)
        right_foot_heading = wrap_to_pi(
            torch.atan2(right_foot_forward[:, 1], right_foot_forward[:, 0]))
        left_foot_heading = wrap_to_pi(
            torch.atan2(left_foot_forward[:, 1], left_foot_forward[:, 0]))
        self.foot_heading[:, 0] = right_foot_heading
        self.foot_heading[:, 1] = left_foot_heading

        self.foot_projected_gravity[:, 0] = quat_rotate_inverse(
            self.foot_states[:, 0, 3:7], self.gravity_vec)
        self.foot_projected_gravity[:, 1] = quat_rotate_inverse(
            self.foot_states[:, 1, 3:7], self.gravity_vec)

        self.update_count += 1
        self.phase_count += 1
        self.phase += 1 / self.full_step_period

        # Ground-truth foot contact
        self.foot_contact = torch.gt(
            self.contact_forces[:, self.feet_ids, 2], 0)
        self.contact_schedule = self.smooth_sqr_wave(self.phase)

        # Update current step from contact
        current_step_masked = self.current_step[self.foot_contact]
        current_step_masked[:, :2] = \
            self.foot_states[self.foot_contact][:, :2]
        current_step_masked[:, 2] = self.foot_heading[self.foot_contact]
        self.current_step[self.foot_contact] = current_step_masked

        naxis = 3
        # BD foot body names for ankle velocity tracking
        self.ankle_vel_history[:, 0, naxis:] = \
            self.ankle_vel_history[:, 0, :naxis]
        self.ankle_vel_history[:, 0, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx[self._RIGHT_FOOT_BODY], 7:10]
        self.ankle_vel_history[:, 1, naxis:] = \
            self.ankle_vel_history[:, 1, :naxis]
        self.ankle_vel_history[:, 1, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx[self._LEFT_FOOT_BODY], 7:10]

    # ------------------------------------------------------------------ #
    # Override foot offset and collision threshold                          #
    # ------------------------------------------------------------------ #
    def _calculate_foot_states(self, foot_states):
        """Adjust foot position by BD-specific ankle-to-contact offset."""
        foot_height_vec = torch.tensor(
            [0., 0., self._FOOT_HEIGHT_OFFSET]
        ).repeat(self.num_envs, 1).to(self.device)

        rfoot_height_vec_world = quat_apply(
            foot_states[:, 0, 3:7], foot_height_vec)
        lfoot_height_vec_world = quat_apply(
            foot_states[:, 1, 3:7], foot_height_vec)
        foot_states[:, 0, :3] += rfoot_height_vec_world
        foot_states[:, 1, :3] += lfoot_height_vec_world
        return foot_states

    # ------------------------------------------------------------------ #
    # Override reset: use BD-appropriate initial stance width              #
    # ------------------------------------------------------------------ #
    def _reset_system(self, env_ids):
        super(HumanoidController, self)._reset_system(env_ids)

        # Foot states
        self.foot_states[env_ids] = self._calculate_foot_states(
            self.rigid_body_state[:, self.feet_ids, :7])[env_ids]
        self.foot_projected_gravity[env_ids, 0] = self.gravity_vec[env_ids]
        self.foot_projected_gravity[env_ids, 1] = self.gravity_vec[env_ids]

        # Initial step commands:
        #   feet_ids[0] = RIGHT foot (R4_Link_ankle) → Y = -hip_half_width
        #   feet_ids[1] = LEFT  foot (L4_Link_ankle) → Y = +hip_half_width
        half_width = 0.054          # half of BD hip-to-hip distance [m]
        self.step_commands[env_ids, 0] = (
            self.env_origins[env_ids]
            + torch.tensor([0., -half_width, 0.], device=self.device))
        self.step_commands[env_ids, 1] = (
            self.env_origins[env_ids]
            + torch.tensor([0.,  half_width, 0.], device=self.device))
        self.foot_on_motion[env_ids, 0] = False
        self.foot_on_motion[env_ids, 1] = True  # left foot starts as swing

        # Step states
        self.current_step[env_ids] = torch.clone(
            self.step_commands[env_ids])
        self.prev_step_commands[env_ids] = torch.clone(
            self.step_commands[env_ids])
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
        self.ICP[env_ids] = 0.
        self.raibert_heuristic[env_ids] = 0.
        self.w[env_ids] = 0.
        self.dstep_length[env_ids] = self.cfg.commands.dstep_length
        self.dstep_width[env_ids] = self.cfg.commands.dstep_width

    # ------------------------------------------------------------------ #
    # Override foot-collision threshold                                     #
    # ------------------------------------------------------------------ #
    def _update_commands(self):
        """Same as parent but with BD-specific foot collision threshold."""
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

        update_step_commands_mask = self.step_commands[
            self.update_commands_ids]
        self.prev_step_commands[self.update_commands_ids] = torch.clone(
            self.step_commands[self.update_commands_ids])

        update_step_commands_mask[
            self.foot_on_motion[self.update_commands_ids]
        ] = self._generate_step_command_by_3DLIPM_XCoM(
            self.update_commands_ids)
        self._update_LIPM_CoM(self.update_commands_ids)

        # BD-specific foot collision threshold (0.10 m vs 0.20 m for MIT)
        foot_collision_ids = (
            update_step_commands_mask[:, 0, :2]
            - update_step_commands_mask[:, 1, :2]
        ).norm(dim=1) < self._FOOT_COLLISION_THRESHOLD

        update_step_commands_mask[foot_collision_ids, :, :2] = (
            self._adjust_foot_collision(
                update_step_commands_mask[foot_collision_ids, :, :2],
                self.foot_on_motion[self.update_commands_ids][
                    foot_collision_ids]))

        if self.cfg.terrain.measure_heights:
            update_step_commands_mask[
                self.foot_on_motion[self.update_commands_ids]
            ] = self._adjust_step_command_in_rough_terrain(
                self.update_commands_ids, update_step_commands_mask)

        self.step_commands[self.update_commands_ids] = (
            update_step_commands_mask)

    # ------------------------------------------------------------------ #
    # Override termination: use BD-appropriate height threshold            #
    # ------------------------------------------------------------------ #
    def check_termination(self):
        """Termination with BD-appropriate thresholds.

        BD is 4.4× lighter than MIT Humanoid → same joint torques produce higher
        angular acceleration → ang_vel threshold raised from 5 → 8 rad/s.
        Height check removed: nominal standing height (0.30 m) equals the
        HumanoidController's threshold, so it would fire during normal walking.
        """
        term_contact = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :],
            dim=-1)
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
