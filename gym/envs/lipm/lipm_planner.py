"""
Linear Inverted Pendulum Model (3D-LIPM) step planner.

Pure pendulum dynamics and XCoM / Raibert step-placement heuristics, kept
separate from any specific robot controller so that multiple environments
(humanoid, BD, ...) can share the same planning layer.

Usage: mix into a LeggedRobot subclass, e.g.

    class BDController(LIPMStepPlanner, LeggedRobot):
        def _init_buffers(self):
            super()._init_buffers()
            self._init_lipm_buffers()

        def _reset_system(self, env_ids):
            super()._reset_system(env_ids)
            self._reset_lipm_buffers(env_ids)

Host requirements (attributes that must exist on `self`):
    Core (from LeggedRobot / cfg / base_task):
      num_envs, device, dt, feet_ids
      sim_params (with .gravity.z)
      rigid_body_state, rigid_body_mass, mass_total
      root_states
      cfg.commands.dstep_width
    Per-step state (populated by the host controller):
      commands, current_step, foot_on_motion
      step_period, update_count, step_stance
      base_lin_vel_world
      right_hip_pos, left_hip_pos

Attributes produced by this mixin (created in _init_lipm_buffers):
    CoM, ICP, w, LIPM_CoM
    support_foot_pos, prev_support_foot_pos
    raibert_heuristic
    step_length, step_width, dstep_length, dstep_width

Reference: Hof, "The extrapolated center of mass concept suggests a simple
control of balance in walking", Hum. Mov. Sci. 27 (2008).
"""

import torch


class LIPMStepPlanner:
    # ------------------------------------------------------------------ #
    # Buffers                                                              #
    # ------------------------------------------------------------------ #
    def _init_lipm_buffers(self):
        """Allocate LIPM state tensors. Call from host _init_buffers()."""
        num_envs = self.num_envs
        num_feet = len(self.feet_ids)
        device = self.device

        self.CoM = torch.zeros(num_envs, 3, dtype=torch.float,
                               device=device, requires_grad=False)
        self.ICP = torch.zeros(num_envs, 3, dtype=torch.float,
                               device=device, requires_grad=False)
        self.w = torch.zeros(num_envs, dtype=torch.float,
                             device=device, requires_grad=False)
        self.LIPM_CoM = torch.zeros(num_envs, 3, dtype=torch.float,
                                    device=device, requires_grad=False)

        self.support_foot_pos = torch.zeros(num_envs, 3, dtype=torch.float,
                                            device=device, requires_grad=False)
        self.prev_support_foot_pos = torch.zeros(num_envs, 3, dtype=torch.float,
                                                 device=device, requires_grad=False)

        self.raibert_heuristic = torch.zeros(num_envs, num_feet, 3,
                                             dtype=torch.float, device=device,
                                             requires_grad=False)

        self.step_length = torch.zeros(num_envs, 1, dtype=torch.float,
                                       device=device, requires_grad=False)
        self.step_width = torch.zeros(num_envs, 1, dtype=torch.float,
                                      device=device, requires_grad=False)
        self.dstep_length = torch.zeros(num_envs, 1, dtype=torch.float,
                                        device=device, requires_grad=False)
        self.dstep_width = torch.zeros(num_envs, 1, dtype=torch.float,
                                       device=device, requires_grad=False)

    def _reset_lipm_buffers(self, env_ids):
        """Reset LIPM state for the given env ids. Call from host _reset_system()."""
        self.ICP[env_ids] = 0.
        self.raibert_heuristic[env_ids] = 0.
        self.w[env_ids] = 0.
        self.dstep_length[env_ids] = self.cfg.commands.dstep_length
        self.dstep_width[env_ids] = self.cfg.commands.dstep_width

    # ------------------------------------------------------------------ #
    # Pendulum state                                                       #
    # ------------------------------------------------------------------ #
    def _calculate_CoM(self):
        """Center of mass from rigid-body state weighted by link masses."""
        self.CoM = (
            self.rigid_body_state[:, :, :3]
            * self.rigid_body_mass.unsqueeze(1)
        ).sum(dim=1) / self.mass_total

    def _calculate_ICP(self):
        """Instantaneous Capture Point.

        x_ic = x + x'/w,  y_ic = y + y'/w,  w = sqrt(g / z_CoM).
        """
        g = -self.sim_params.gravity.z
        self.w = torch.sqrt(g / self.CoM[:, 2:3])
        self.ICP[:, :2] = self.CoM[:, :2] + self.root_states[:, 7:9] / self.w

    def _update_LIPM_CoM(self, update_commands_ids):
        """Propagate LIPM CoM one dt forward under the pendulum dynamics."""
        self.LIPM_CoM[update_commands_ids] = self.CoM[update_commands_ids].clone()

        T = self.dt
        g = -self.sim_params.gravity.z
        w = torch.sqrt(g / self.LIPM_CoM[:, 2:3])

        right_step_ids = torch.where(torch.where(self.foot_on_motion)[1] == 0)[0]
        left_step_ids = torch.where(torch.where(self.foot_on_motion)[1] == 1)[0]

        support_foot_pos = self.support_foot_pos.clone()
        support_foot_pos[right_step_ids] = self.current_step[right_step_ids, 1, :3]
        support_foot_pos[left_step_ids] = self.current_step[left_step_ids, 0, :3]

        x_0 = self.LIPM_CoM[:, 0:1] - support_foot_pos[:, 0:1]
        y_0 = self.LIPM_CoM[:, 1:2] - support_foot_pos[:, 1:2]
        vx_0 = self.root_states[:, 7:8]
        vy_0 = self.root_states[:, 8:9]

        x_f = x_0 * torch.cosh(T * w) + vx_0 * torch.sinh(T * w) / w
        y_f = y_0 * torch.cosh(T * w) + vy_0 * torch.sinh(T * w) / w

        self.LIPM_CoM[:, 0:1] = x_f + support_foot_pos[:, 0:1]
        self.LIPM_CoM[:, 1:2] = y_f + support_foot_pos[:, 1:2]

    # ------------------------------------------------------------------ #
    # Raibert heuristic                                                    #
    # ------------------------------------------------------------------ #
    def _calculate_raibert_heuristic(self):
        """Raibert step-location heuristic: r = p_hip + p_symmetry.

        p_symmetry = 0.5 * t_stance * v + k * (v - v_cmd),   k = sqrt(h / g)
        """
        g = -self.sim_params.gravity.z
        k = torch.sqrt(self.CoM[:, 2:3] / g)
        p_symmetry = (
            0.5 * self.step_stance * self.dt * self.base_lin_vel_world[:, :2]
            + k * (self.base_lin_vel_world[:, :2] - self.commands[:, :2]))
        self.raibert_heuristic[:, 0, :2] = self.right_hip_pos[:, :2] + p_symmetry
        self.raibert_heuristic[:, 1, :2] = self.left_hip_pos[:, :2] + p_symmetry

    def _generate_step_command_by_raibert_heuristic(self, update_commands_ids):
        """Step command from the precomputed Raibert heuristic."""
        foot_on_motion = self.foot_on_motion[update_commands_ids]
        commands = self.commands[update_commands_ids]
        raibert_heuristic = self.raibert_heuristic[update_commands_ids]
        theta = torch.atan2(commands[:, 1:2], commands[:, 0:1])

        right_step_ids = torch.where(torch.where(foot_on_motion)[1] == 0)[0]
        left_step_ids = torch.where(torch.where(foot_on_motion)[1] == 1)[0]

        random_step_command = torch.zeros(foot_on_motion.sum(), 3,
                                          dtype=torch.float, device=self.device,
                                          requires_grad=False)
        random_step_command[right_step_ids, :2] = raibert_heuristic[right_step_ids, 0, :2]
        random_step_command[left_step_ids, :2] = raibert_heuristic[left_step_ids, 1, :2]
        random_step_command[:, 2] = theta.squeeze(1)
        return random_step_command

    def _generate_dynamic_step_command_by_raibert_heuristic(self, update_commands_ids):
        """Raibert heuristic recomputed with remaining horizon T_remaining."""
        foot_on_motion = self.foot_on_motion[update_commands_ids]
        commands = self.commands[update_commands_ids]
        theta = torch.atan2(commands[:, 1:2], commands[:, 0:1])
        step_period = self.step_period[update_commands_ids]
        update_count = self.update_count[update_commands_ids]
        T = (step_period - update_count.unsqueeze(1)) * self.dt

        g = -self.sim_params.gravity.z
        k = torch.sqrt(self.CoM[:, 2:3] / g)
        p_symmetry = (
            0.5 * T * self.base_lin_vel_world[:, :2]
            + k * (self.base_lin_vel_world[:, :2] - self.commands[:, :2]))

        dynamic_raibert_heuristic = torch.zeros(
            self.num_envs, len(self.feet_ids), 3,
            dtype=torch.float, device=self.device, requires_grad=False)
        dynamic_raibert_heuristic[:, 0, :2] = self.right_hip_pos[:, :2] + p_symmetry
        dynamic_raibert_heuristic[:, 1, :2] = self.left_hip_pos[:, :2] + p_symmetry
        raibert_heuristic = dynamic_raibert_heuristic[update_commands_ids]

        right_step_ids = torch.where(torch.where(foot_on_motion)[1] == 0)[0]
        left_step_ids = torch.where(torch.where(foot_on_motion)[1] == 1)[0]

        random_step_command = torch.zeros(foot_on_motion.sum(), 3,
                                          dtype=torch.float, device=self.device,
                                          requires_grad=False)
        random_step_command[right_step_ids, :2] = raibert_heuristic[right_step_ids, 0, :2]
        random_step_command[left_step_ids, :2] = raibert_heuristic[left_step_ids, 1, :2]
        random_step_command[:, 2] = theta.squeeze(1)
        return random_step_command

    # ------------------------------------------------------------------ #
    # 3D-LIPM XCoM step planning                                           #
    # ------------------------------------------------------------------ #
    def _generate_step_command_by_3DLIPM_XCoM(self, update_commands_ids):
        """XCoM step placement with fixed planning horizon T = step_period * dt."""
        T = self.step_period[update_commands_ids] * self.dt
        dstep_width = self.dstep_width[update_commands_ids]
        return self._xcom_step_command(update_commands_ids, T, dstep_width)

    def _generate_dynamic_step_command_by_3DLIPM_XCoM(self, update_commands_ids):
        """XCoM step placement recomputed each tick with shrinking horizon T_remaining."""
        step_period = self.step_period[update_commands_ids]
        update_count = self.update_count[update_commands_ids]
        T = (step_period - update_count.unsqueeze(1)) * self.dt
        dstep_width = self.cfg.commands.dstep_width * T / (step_period * self.dt)
        return self._xcom_step_command(update_commands_ids, T, dstep_width)

    def _xcom_step_command(self, update_commands_ids, T, dstep_width):
        """Core XCoM step computation shared by the static and dynamic variants.

        T:           planning horizon [m x 1]  (either full step or remaining)
        dstep_width: desired lateral step width [m x 1]
        """
        foot_on_motion = self.foot_on_motion[update_commands_ids]
        commands = self.commands[update_commands_ids]
        current_step = self.current_step[update_commands_ids]
        CoM = self.CoM[update_commands_ids]
        w = self.w[update_commands_ids]
        dstep_length = torch.norm(commands[:, :2], dim=1, keepdim=True) * T
        theta = torch.atan2(commands[:, 1:2], commands[:, 0:1])

        right_step_ids = torch.where(torch.where(foot_on_motion)[1] == 0)[0]
        left_step_ids = torch.where(torch.where(foot_on_motion)[1] == 1)[0]

        root_states = self.root_states[update_commands_ids]
        support_foot_pos = self.support_foot_pos[update_commands_ids]
        support_foot_pos[right_step_ids] = current_step[right_step_ids, 1, :3]
        support_foot_pos[left_step_ids] = current_step[left_step_ids, 0, :3]

        # Logging: actual step_length / width in body-heading frame
        rright_x = (torch.cos(theta) * current_step[:, 0, 0:1]
                    + torch.sin(theta) * current_step[:, 0, 1:2])
        rright_y = (-torch.sin(theta) * current_step[:, 0, 0:1]
                    + torch.cos(theta) * current_step[:, 0, 1:2])
        rleft_x = (torch.cos(theta) * current_step[:, 1, 0:1]
                   + torch.sin(theta) * current_step[:, 1, 1:2])
        rleft_y = (-torch.sin(theta) * current_step[:, 1, 0:1]
                   + torch.cos(theta) * current_step[:, 1, 1:2])
        self.step_length[update_commands_ids] = torch.abs(rright_x - rleft_x)
        self.step_width[update_commands_ids] = torch.abs(rright_y - rleft_y)
        self.dstep_length[update_commands_ids] = dstep_length
        self.dstep_width[update_commands_ids] = dstep_width

        # LIPM forward integration of CoM state over horizon T
        x_0 = CoM[:, 0:1] - support_foot_pos[:, 0:1]
        y_0 = CoM[:, 1:2] - support_foot_pos[:, 1:2]
        vx_0 = root_states[:, 7:8]
        vy_0 = root_states[:, 8:9]

        x_f = x_0 * torch.cosh(T * w) + vx_0 * torch.sinh(T * w) / w
        vx_f = x_0 * w * torch.sinh(T * w) + vx_0 * torch.cosh(T * w)
        y_f = y_0 * torch.cosh(T * w) + vy_0 * torch.sinh(T * w) / w
        vy_f = y_0 * w * torch.sinh(T * w) + vy_0 * torch.cosh(T * w)

        x_f_world = x_f + support_foot_pos[:, 0:1]
        y_f_world = y_f + support_foot_pos[:, 1:2]

        # Extrapolated ICP at touchdown
        eICP_x = x_f_world + vx_f / w
        eICP_y = y_f_world + vy_f / w

        # XCoM offsets (Hof 2008): b = L / (e^{Tw} ± 1)
        b_x = dstep_length / (torch.exp(T * w) - 1)
        b_y = dstep_width / (torch.exp(T * w) + 1)

        original_offset_x = -b_x
        original_offset_y = -b_y
        original_offset_y[left_step_ids] = b_y[left_step_ids]

        # Rotate offsets from heading-aligned frame into world frame
        offset_x = torch.cos(theta) * original_offset_x - torch.sin(theta) * original_offset_y
        offset_y = torch.sin(theta) * original_offset_x + torch.cos(theta) * original_offset_y

        random_step_command = torch.zeros(foot_on_motion.sum(), 3,
                                          dtype=torch.float, device=self.device,
                                          requires_grad=False)
        random_step_command[:, 0] = (eICP_x + offset_x).squeeze(1)
        random_step_command[:, 1] = (eICP_y + offset_y).squeeze(1)
        random_step_command[:, 2] = theta.squeeze(1)
        return random_step_command
