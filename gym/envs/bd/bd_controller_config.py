"""
Configuration for BD bipedal robot (8.42 kg, 10 DOF).
URDF: assets/BD.urdf

BD kinematic structure (L = Left, R = Right):
  base_link
  ├── J_L0 (hip yaw,   axis +Z) → L0_Link
  │   └── J_L1 (hip roll,  axis -X) → L1_Link
  │       └── J_L2 (hip pitch, axis +Y) → L2_Link
  │           └── J_L3 (knee,      axis +Y) → L3_Link
  │               └── J_L4_ankle (ankle, axis -Y) → L4_Link_ankle  [LEFT FOOT]
  └── J_R0 (hip yaw,   axis +Z) → R0_Link
      └── J_R1 (hip roll,  axis -X) → R1_Link
          └── J_R2 (hip pitch, axis -Y) → R2_Link
              └── J_R3 (knee,      axis -Y) → R3_Link
                  └── J_R4_ankle (ankle, axis +Y) → R4_Link_ankle  [RIGHT FOOT]

NOTE on axis symmetry:
  Left  leg uses +Y for hip pitch and knee → negative = flexion
  Right leg uses -Y for hip pitch and knee → positive = same physical flexion
  Left  ankle uses -Y → positive = dorsiflexion
  Right ankle uses +Y → negative = same physical dorsiflexion

DOF order in IsaacGym (follows URDF joint declaration order):
  idx 0: J_L0  (left  hip yaw)
  idx 1: J_L1  (left  hip roll)
  idx 2: J_L2  (left  hip pitch)
  idx 3: J_L3  (left  knee)
  idx 4: J_L4_ankle (left  ankle)
  idx 5: J_R0  (right hip yaw)
  idx 6: J_R1  (right hip roll)
  idx 7: J_R2  (right hip pitch)
  idx 8: J_R3  (right knee)
  idx 9: J_R4_ankle (right ankle)

Robot parameters:
  Total mass:      8.42 kg
  Straight-leg height: 0.342 m (base_link to ground at zero joint angles)
  Target stand height: 0.32 m (with slight knee bend)
  Hip-to-hip width:    0.107 m
"""

import torch
from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotRunnerCfg


class BDControllerCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        # num_envs = 4096
        num_envs = 4096
        num_actuators = 10
        episode_length_s = 10

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane'
        measure_heights = False
        selected = True
        terrain_kwargs = {'type': 'stepping_stones'}
        difficulty = 5.0
        terrain_length = 18.
        terrain_width = 18.
        terrain_proportions = [0., 0.5, 0., 0.5, 0., 0., 0.]

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = 'reset_to_basic'
        pos = [0., 0., 0.38]          # x,y,z [m] — box foot bottom at ~0.341 m; 4 mm clearance
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        # Nominal standing pose.
        # Left leg:  axis +Y for hip-pitch/knee → negative = flexion
        # Right leg: axis -Y for hip-pitch/knee → positive = flexion (mirrored)
        # Left ankle:  axis -Y → positive = dorsiflexion
        # Right ankle: axis +Y → negative = dorsiflexion
        # default_joint_angles = {
        #     'J_L0':       0.0,     # left  hip yaw
        #     'J_L1':       0.05,    # left  hip roll — slight abduction
        #     'J_L2':       0.3,     # left  hip pitch — slight flexion (+Y axis)
        #     'J_L3':      -0.6,     # left  knee — bent (+Y axis, negative = flex)
        #     'J_L4_ankle': 0.3,     # left  ankle — dorsiflexion (-Y axis, positive = DF)
        #     'J_R0':       0.0,     # right hip yaw
        #     'J_R1':       0.05,    # right hip roll — slight abduction
        #     'J_R2':      -0.3,     # right hip pitch — same flex (-Y axis, negative = flex)
        #     'J_R3':       0.6,     # right knee — bent (-Y axis, positive = flex)
        #     'J_R4_ankle': -0.3,    # right ankle — dorsiflexion (+Y axis, negative = DF)
        # }
        default_joint_angles = {
            "J_L0":   0.0,
            "J_L1":  0.08,
            "J_L2":  0.56,
            "J_L3":  -1.12,
            "J_L4_ankle": -0.57,

            "J_R0":   0.0,
            "J_R1":  -0.08,
            "J_R2":  -0.56,
            "J_R3":  1.12,
            "J_R4_ankle": 0.57
        }

        # Reset range for 'reset_to_range' mode — stays within joint limits
        root_pos_range = [
            [0., 0.],              # x
            [0., 0.],              # y
            [0.34, 0.37],          # z  (box foot bottom ~0.341 m; 4 mm clearance)
            [-torch.pi/12, torch.pi/12],   # roll
            [-torch.pi/12, torch.pi/12],   # pitch
            [-torch.pi/12, torch.pi/12],   # yaw
        ]
        root_vel_range = [
            [-.3, .3], [-.3, .3], [-.3, .3],
            [-.3, .3], [-.3, .3], [-.3, .3],
        ]

        dof_pos_range = {
            'J_L0':       [-0.1,  0.1],
            'J_L1':       [ 0.0,  0.1],
            'J_L2':       [ 0.2,  0.4],
            'J_L3':       [-0.7, -0.5],
            'J_L4_ankle': [ 0.2,  0.4],
            'J_R0':       [-0.1,  0.1],
            'J_R1':       [ 0.0,  0.1],
            'J_R2':       [-0.4, -0.2],
            'J_R3':       [ 0.5,  0.7],
            'J_R4_ankle': [-0.4, -0.2],
        }

        dof_vel_range = {
            'J_L0':       [-0.1, 0.1],
            'J_L1':       [-0.1, 0.1],
            'J_L2':       [-0.1, 0.1],
            'J_L3':       [-0.1, 0.1],
            'J_L4_ankle': [-0.1, 0.1],
            'J_R0':       [-0.1, 0.1],
            'J_R1':       [-0.1, 0.1],
            'J_R2':       [-0.1, 0.1],
            'J_R3':       [-0.1, 0.1],
            'J_R4_ankle': [-0.1, 0.1],
        }

    class control(LeggedRobotCfg.control):
        # PD gains scaled from MIT Humanoid (Kp=30→~7, rounded to 10)
        # BD mass ratio: 8.42/37 ≈ 0.23, torque limits 12-20 Nm
        # stiffness = {
        #     'J_L0':       10.,   # hip yaw   (effort 12 Nm)
        #     'J_L1':       15.,   # hip roll  (effort 20 Nm)
        #     'J_L2':       15.,   # hip pitch (effort 20 Nm)
        #     'J_L3':       20.,   # knee      (effort 20 Nm)
        #     'J_L4_ankle': 10.,   # ankle     (effort 12 Nm)
        #     'J_R0':       10.,
        #     'J_R1':       15.,
        #     'J_R2':       15.,
        #     'J_R3':       20.,
        #     'J_R4_ankle': 10.,
        # }
        stiffness = {
            'J_L0':       25.,   # hip yaw   (effort 12 Nm)
            'J_L1':       25.,   # hip roll  (effort 20 Nm)
            'J_L2':       25.,   # hip pitch (effort 20 Nm)
            'J_L3':       25.,   # knee      (effort 20 Nm)
            'J_L4_ankle': 25.,   # ankle     (effort 12 Nm)
            'J_R0':       25.,
            'J_R1':       25.,
            'J_R2':       25.,
            'J_R3':       25.,
            'J_R4_ankle': 2.,
        }
        damping = {
            'J_L0':       0.5,
            'J_L1':       0.5,
            'J_L2':       0.5,
            'J_L3':       0.5,
            'J_L4_ankle': 0.5,
            'J_R0':       0.5,
            'J_R1':       0.5,
            'J_R2':       0.5,
            'J_R3':       0.5,
            'J_R4_ankle': 0.5,
        }
        actuation_scale = 1.0
        exp_avg_decay = None
        decimation = 10            # 100 Hz control, 1 kHz physics

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 3
        resampling_time = 10.

        succeed_step_radius = 0.03
        succeed_step_angle = 10
        apex_height_percentage = 0.15

        sample_angle_offset = 20
        sample_radius_offset = 0.05

        # BD step geometry: shorter stride due to smaller legs (~31 cm effective length)
        dstep_length = 0.07        # nominal step length [m]
        dstep_width  = 0.24         # nominal step width  [m]  ≈ hip width

        class ranges(LeggedRobotCfg.commands.ranges):
            # BD CoM height ~0.30m → w=5.72 → need T≈0.245s to match MIT T*w≈1.4
            # MIT used 35 steps @ 0.01s=0.35s with w=3.88 → T*w=1.36
            # BD:  24 steps @ 0.01s=0.24s with w=5.72 → T*w=1.37  ✓
            sample_period = [24, 26]        # gait frequency ≈ 4.2 Hz
            dstep_width = [0.22, 0.26]      # [m]
            dstep_length = [0.06, 0.08]
            

            lin_vel_x = [0.2, 0.3]         # [m/s]
            lin_vel_y = 0.01                  # [m/s]
            yaw_vel   = 0.                   # [rad/s]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]

        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]      # ±0.5 kg on 8.42 kg ≈ ±6%

        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.3               # smaller robot → smaller impulses

    class asset(LeggedRobotCfg.asset):
        # URDF lives in assets/ next to the repo root; meshes at assets/meshes/
        file = '{LEGGED_GYM_ROOT_DIR}/assets/BD.urdf'

        # end_effectors ORDER MATTERS: index 0 = RIGHT foot, index 1 = LEFT foot
        # (matches the convention used throughout HumanoidController)
        end_effectors = ['R4_Link_ankle', 'L4_Link_ankle']
        foot_name = 'ankle'               # substring fallback (not used if end_effectors drives feet_ids)
        keypoints = ['base_link']

        terminate_after_contacts_on = [
            'base_link',
            'L2_Link',   # left  thigh
            'L3_Link',   # left  shin
            'R2_Link',   # right thigh
            'R3_Link',   # right shin
        ]

        disable_gravity = False
        disable_actuations = False
        disable_motors = False

        self_collisions = 1    # 1 = disable self-collisions (prevents mesh overlap at spawn)
        collapse_fixed_joints = False
        flip_visual_attachments = False
        default_dof_drive_mode = 3         # effort mode

        angular_damping = 0.05             # global joint damping (smaller robot)

        # Rotor inertia (armature) for each DOF, order = URDF joint order:
        # [J_L0, J_L1, J_L2, J_L3, J_L4_ankle,  J_R0, J_R1, J_R2, J_R3, J_R4_ankle]
        rotor_inertia = [
            0.001, 0.001, 0.002, 0.004, 0.001,    # LEFT  leg
            0.001, 0.001, 0.002, 0.004, 0.001,    # RIGHT leg
        ]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.2          # target standing height [m]
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 500.            # smaller robot → lower max force

        curriculum = False
        only_positive_rewards = False
        tracking_sigma = 0.25

        class weights(LeggedRobotCfg.rewards.weights):
            # Regularisation
            actuation_rate  = 1e-3
            actuation_rate2 = 1e-4
            torques         = 1e-4
            dof_vel         = 1e-3
            lin_vel_z       = 1e-1
            ang_vel_xy      = 1e-2
            dof_pos_limits  = 10.
            torque_limits   = 1e-2

            # Floating base
            base_height        = 2.
            base_heading       = 4.
            base_z_orientation = 3.
            tracking_lin_vel_world = 5.

            # Stepping
            joint_regularization = 1.
            contact_schedule     = 3.

        class termination_weights(LeggedRobotCfg.rewards.termination_weights):
            termination = 1.

    class scaling(LeggedRobotCfg.scaling):
        base_height       = 1.
        base_lin_vel      = 1.
        base_ang_vel      = 1.
        projected_gravity = 1.
        foot_states_right = 1.
        foot_states_left  = 1.
        dof_pos           = 1.
        dof_vel           = 1.
        dof_pos_target    = 1.
        commands          = 1.
        clip_actions      = 1.0            # BD joint limits are small (hip yaw ±0.3 rad)


class BDControllerRunnerCfg(LeggedRobotRunnerCfg):
    do_wandb = True
    seed = -1

    class policy(LeggedRobotRunnerCfg.policy):
        init_noise_std = 0.1           # small initial exploration noise; Xavier output dominates at 1.0
        actor_hidden_dims  = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        activation = 'elu'
        normalize_obs = True
        last_layer_gain = 0.01         # scale last layer to near-zero initial actions

        # Observation vector: same 51-dim structure as MIT Humanoid
        # 1 + 3 + 1 + 3 + 3 + 4 + 4 + 4 + 4 + 3 + 1 + 1 + 10 + 10 = 51
        actor_obs = [
            "base_heading",          # 1
            "base_ang_vel",          # 3
            "projected_gravity",     # 3
            "foot_states_right",     # 4
            "foot_states_left",      # 4
            "step_commands_right",   # 4
            "step_commands_left",    # 4
            "commands",              # 3
            "phase_sin",             # 1
            "phase_cos",             # 1
            "dof_pos",               # 10
            "dof_vel",               # 10
        ]
        critic_obs = [
            "base_height",           # 1
            "base_lin_vel_world",    # 3
            "base_heading",          # 1
            "base_ang_vel",          # 3
            "projected_gravity",     # 3
            "foot_states_right",     # 4
            "foot_states_left",      # 4
            "step_commands_right",   # 4
            "step_commands_left",    # 4
            "commands",              # 3
            "phase_sin",             # 1
            "phase_cos",             # 1
            "dof_pos",               # 10
            "dof_vel",               # 10
        ]
        actions = ["dof_pos_target"]

        class noise:
            base_height          = 0.05
            base_lin_vel         = 0.05
            base_lin_vel_world   = 0.05
            base_heading         = 0.01
            base_ang_vel         = 0.05
            projected_gravity    = 0.05
            foot_states_right    = 0.01
            foot_states_left     = 0.01
            step_commands_right  = 0.05
            step_commands_left   = 0.05
            commands             = 0.1
            dof_pos              = 0.05
            dof_vel              = 0.5
            foot_contact         = 0.1

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        class PPO:
            value_loss_coef      = 1.0
            use_clipped_value_loss = True
            clip_param           = 0.2
            entropy_coef         = 0.01
            num_learning_epochs  = 5
            num_mini_batches     = 4
            learning_rate        = 1e-5
            schedule             = 'adaptive'
            gamma                = 0.99
            lam                  = 0.95
            desired_kl           = 0.01
            max_grad_norm        = 1.

    class runner(LeggedRobotRunnerCfg.runner):
        policy_class_name    = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env    = 24
        max_iterations       = 5000
        run_name             = 'bd'
        experiment_name      = 'BD_Controller'
        save_interval        = 100
        plot_input_gradients = False
        plot_parameter_gradients = False
