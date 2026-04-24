"""
BD viewer overlays (velocity arrows, footsteps, step commands).

Mixin invoked each frame via `_visualization()`. All drawing happens through
IsaacGym's gymutil wireframe helpers; no effect in headless mode.

Expected host attributes:
    num_envs, envs, gym, viewer, device
    base_pos, base_quat
    commands
    current_step, step_commands
"""

import torch
from isaacgym import gymutil
from isaacgym.torch_utils import quat_apply

from gym.envs.humanoid.humanoid_utils import FootStepGeometry, VelCommandGeometry


class BDVisualization:
    def _visualization(self):
        """Entry point called by LeggedRobot every rendered frame."""
        self.gym.clear_lines(self.viewer)
        self._draw_world_velocity_arrow_vis()
        self._draw_step_vis()
        self._draw_step_command_vis()

    def _draw_velocity_arrow_vis(self):
        """Linear + angular velocity command arrows in base frame."""
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0., 0., .5]).repeat(self.num_envs, 1).to(self.device))
        lin_vel_command = quat_apply(
            self.base_quat,
            torch.cat((self.commands[:, :2],
                       torch.zeros((self.num_envs, 1), device=self.device)), dim=1) / 5)
        ang_vel_command = quat_apply(
            self.base_quat,
            torch.cat((torch.zeros((self.num_envs, 2), device=self.device),
                       self.commands[:, 2:3]), dim=1) / 5)
        for i in range(self.num_envs):
            lin_arrow = VelCommandGeometry(origins[i], lin_vel_command[i], color=(0, 1, 0))
            ang_arrow = VelCommandGeometry(origins[i], ang_vel_command[i], color=(0, 1, 0))
            gymutil.draw_lines(lin_arrow, self.gym, self.viewer, self.envs[i], pose=None)
            gymutil.draw_lines(ang_arrow, self.gym, self.viewer, self.envs[i], pose=None)

    def _draw_world_velocity_arrow_vis(self):
        """Linear velocity command arrow in world frame."""
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0., 0., .5]).repeat(self.num_envs, 1).to(self.device))
        lin_vel_command = torch.cat(
            (self.commands[:, :2],
             torch.zeros((self.num_envs, 1), device=self.device)), dim=1) / 5
        for i in range(self.num_envs):
            lin_arrow = VelCommandGeometry(origins[i], lin_vel_command[i], color=(0, 1, 0))
            gymutil.draw_lines(lin_arrow, self.gym, self.viewer, self.envs[i], pose=None)

    def _draw_step_vis(self):
        """Last landed footsteps (right = magenta, left = cyan)."""
        for i in range(self.num_envs):
            right_step = FootStepGeometry(
                self.current_step[i, 0, :2], self.current_step[i, 0, 2], color=(1, 0, 1))
            left_step = FootStepGeometry(
                self.current_step[i, 1, :2], self.current_step[i, 1, 2], color=(0, 1, 1))
            gymutil.draw_lines(left_step, self.gym, self.viewer, self.envs[i], pose=None)
            gymutil.draw_lines(right_step, self.gym, self.viewer, self.envs[i], pose=None)

    def _draw_step_command_vis(self):
        """Planned next footstep targets (right = red, left = blue)."""
        for i in range(self.num_envs):
            right_cmd = FootStepGeometry(
                self.step_commands[i, 0, :2], self.step_commands[i, 0, 2], color=(1, 0, 0))
            left_cmd = FootStepGeometry(
                self.step_commands[i, 1, :2], self.step_commands[i, 1, 2], color=(0, 0, 1))
            gymutil.draw_lines(left_cmd, self.gym, self.viewer, self.envs[i], pose=None)
            gymutil.draw_lines(right_cmd, self.gym, self.viewer, self.envs[i], pose=None)
