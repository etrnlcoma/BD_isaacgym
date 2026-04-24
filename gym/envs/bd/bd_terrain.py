"""
BD terrain-aware step command adjustments.

Mixin that warps the next planned footstep based on the measured heightmap
around the robot:
  * rough terrain → lift step target to the local ground height
  * gap terrain   → slide step target off gap edges to the nearest flat patch

Active only when `cfg.terrain.measure_heights = True`.

Expected host attributes:
    num_envs, device, num_height_points
    base_quat, base_pos
    height_points, measured_heights
    foot_on_motion, current_step
    terrain (with cfg.measured_points_x_num_sample / _y_num_sample)
"""

import torch
import torch.nn.functional as F

from gym.envs.humanoid.humanoid_utils import smart_sort
from gym.utils.math import quat_apply_yaw


class BDTerrainAdapter:
    def _adjust_step_command_in_rough_terrain(self, update_commands_ids,
                                              update_step_commands_mask):
        """Lift step target z to the ground height at the planned (x, y)."""
        step_command_mask = update_step_commands_mask[
            self.foot_on_motion[update_commands_ids]]
        height_points = quat_apply_yaw(
            self.base_quat[update_commands_ids].repeat(1, self.num_height_points),
            self.height_points[update_commands_ids])
        height_points[:, :, 0] += self.base_pos[update_commands_ids, 0:1]
        height_points[:, :, 1] += self.base_pos[update_commands_ids, 1:2]

        measured_heights_mask = self.measured_heights[update_commands_ids]
        nearest_hpts_idx = torch.argmin(
            torch.norm(height_points[:, :, :2] - step_command_mask[:, None, :2],
                       dim=2), dim=1)
        step_command_mask[:, 2] = measured_heights_mask[
            torch.arange(len(nearest_hpts_idx)), nearest_hpts_idx]
        return step_command_mask

    def _adjust_step_command_in_gap_terrain(self, update_commands_ids,
                                            update_step_commands_mask):
        """Slide step target off gap edges to the nearest flat patch."""
        step_command_mask = update_step_commands_mask[
            self.foot_on_motion[update_commands_ids]]
        support_step_masked = self.current_step[~self.foot_on_motion][update_commands_ids]

        height_points = quat_apply_yaw(
            self.base_quat[update_commands_ids].repeat(1, self.num_height_points),
            self.height_points[update_commands_ids])
        height_points[:, :, 0] += self.base_pos[update_commands_ids, 0:1]
        height_points[:, :, 1] += self.base_pos[update_commands_ids, 1:2]

        # Mask out support foot cell so the swing foot doesn't land on it.
        measured_heights_mask = self.measured_heights[update_commands_ids]
        nearest_hpts_idx = torch.argmin(
            torch.norm(height_points[:, :, :2] - support_step_masked[:, None, :2],
                       dim=2), dim=1)
        measured_heights_mask[torch.arange(len(nearest_hpts_idx)), nearest_hpts_idx] = -10.
        self.measured_heights[update_commands_ids] = measured_heights_mask

        # Smooth heightmap (5x3 box kernel) to avoid targeting single-cell ridges.
        avg_heights = self.measured_heights.reshape(
            self.num_envs, 1,
            self.terrain.cfg.measured_points_x_num_sample,
            self.terrain.cfg.measured_points_y_num_sample)
        kernel = (1 / 15) * torch.ones((1, 1, 5, 3), device=self.device)
        avg_heights = F.conv2d(avg_heights, kernel, padding='same').reshape(
            self.num_envs, -1)
        height_points[:, :, 2] = avg_heights[update_commands_ids]

        # For each planned step, if nearest height-point is in a gap (z < -0.1),
        # snap to the closest point that is on a flat surface.
        values, indices = torch.norm(
            height_points[:, :, :2] - step_command_mask[:, None, :2], dim=2).sort()
        nearest_hpts_idx = indices[:, 0]
        nearest_hpts_height = height_points[
            torch.arange(len(nearest_hpts_idx)), nearest_hpts_idx, 2]
        modify_steps_idx = nearest_hpts_height < -0.1
        closest_flat_hpts_idx = (
            smart_sort(height_points[:, :, 2], indices) > -0.1).int().argmax(dim=1)
        closest_flat_hpts_idx = indices[
            torch.arange(len(closest_flat_hpts_idx)), closest_flat_hpts_idx]
        step_command_mask[modify_steps_idx, :2] = height_points[
            torch.arange(len(closest_flat_hpts_idx)),
            closest_flat_hpts_idx, :2][modify_steps_idx]
        return step_command_mask
