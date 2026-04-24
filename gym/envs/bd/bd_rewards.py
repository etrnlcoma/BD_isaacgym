"""
BD reward terms.

Mixin that bundles all `_reward_*` methods used by BDController. Each reward
is registered automatically by LeggedRobot through the `weights` section of
BDControllerCfg.rewards — names must match `_reward_<name>` ↔ `weights.<name>`.

Expected host attributes (from LeggedRobot / BDController):
    base_height, base_heading, base_lin_vel_world
    projected_gravity
    commands, root_states
    dof_pos, scales
    foot_contact, foot_on_motion
    step_location_offset, contact_schedule
    cfg.rewards.base_height_target
    _neg_exp(), _negsqrd_exp() — helpers from LeggedRobot
"""

import torch

from gym.utils.math import wrap_to_pi


class BDRewards:
    # * Floating base rewards * #
    def _reward_base_height(self):
        """Track specified standing height."""
        error = (self.cfg.rewards.base_height_target - self.base_height).flatten()
        return self._negsqrd_exp(error)

    def _reward_base_heading(self):
        """Align base heading with commanded velocity direction."""
        command_heading = torch.atan2(self.commands[:, 1], self.commands[:, 0])
        base_heading_error = torch.abs(
            wrap_to_pi(command_heading - self.base_heading.squeeze(1)))
        return self._neg_exp(base_heading_error, a=torch.pi / 2)

    def _reward_base_z_orientation(self):
        """Reward upright base orientation."""
        error = torch.norm(self.projected_gravity[:, :2], dim=1)
        return self._negsqrd_exp(error, a=0.2)

    def _reward_tracking_lin_vel_world(self):
        """Track linear velocity command in world frame."""
        error = self.commands[:, :2] - self.root_states[:, 7:9]
        error *= 1. / (1. + torch.abs(self.commands[:, :2]))
        return self._negsqrd_exp(error, a=1.).sum(dim=1)

    # * Stepping rewards * #
    def _reward_joint_regularization(self):
        """Regularize hip-yaw and hip-roll around zero.

        BD DOF order: [J_L0, J_L1, J_L2, J_L3, J_L4_ankle,
                       J_R0, J_R1, J_R2, J_R3, J_R4_ankle]
        Indices 0, 5 = yaw (L0, R0); indices 1, 6 = roll (L1, R1).
        """
        error = 0.
        error += self._negsqrd_exp(self.dof_pos[:, 0] / self.scales['dof_pos'])
        error += self._negsqrd_exp(self.dof_pos[:, 5] / self.scales['dof_pos'])
        error += self._negsqrd_exp(self.dof_pos[:, 1] / self.scales['dof_pos'])
        error += self._negsqrd_exp(self.dof_pos[:, 6] / self.scales['dof_pos'])
        return error / 4

    def _reward_contact_schedule(self):
        """Alternating right/left contacts, gated by step-tracking error."""
        contact_rewards = (
            (self.foot_contact[:, 0].int() - self.foot_contact[:, 1].int())
            * self.contact_schedule.squeeze(1))
        k, a = 3., 1.
        tracking_rewards = k * self._neg_exp(
            self.step_location_offset[~self.foot_on_motion], a=a)
        return contact_rewards * tracking_rewards
