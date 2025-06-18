import gc
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import hydra
from omegaconf import DictConfig, OmegaConf
from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from pyrep.const import RenderMode

from roboprompt_agent import RoboPromptAgent
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.utils.rollout_generator import RolloutGenerator

from utils import CAMERAS, SCENE_BOUNDS, ROTATION_RESOLUTION, VOXEL_SIZE, IMAGE_SIZE

import torch
from torch.multiprocessing import Manager
torch.multiprocessing.set_sharing_strategy('file_system')

def create_obs_config():
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=True,
        depth=False,
        image_size=IMAGE_SIZE,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in CAMERAS:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config

def eval_seed(eval_cfg,
              logdir,
              cams,
              env_device,
              multi_task,
              seed,
              env_config) -> None:

    tasks = eval_cfg.rlbench.tasks
    rg = RolloutGenerator()

    agent = RoboPromptAgent(eval_cfg.rlbench.task_name, eval_cfg.model)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    # make the directory first so that the weightsdir is created
    # we don't actually load the weights here
    os.makedirs(eval_cfg.framework.logdir, exist_ok=True)
    env_runner = IndependentEnvRunner(
        train_env=None,
        agent=agent,
        train_replay_buffer=None,
        num_train_envs=0,
        num_eval_envs=eval_cfg.framework.eval_envs,
        rollout_episodes=99999,
        eval_episodes=eval_cfg.framework.eval_episodes,
        training_iterations=0,
        eval_from_eps_number=eval_cfg.framework.eval_from_eps_number,
        episode_length=eval_cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=eval_cfg.framework.logdir,
        logdir=logdir,
        env_device=env_device,
        rollout_generator=rg,
        num_eval_runs=len(tasks),
        multi_task=multi_task)

    manager = Manager()
    save_load_lock = manager.Lock()
    writer_lock = manager.Lock()
    
    env_runner.start({"task": 0}, save_load_lock, writer_lock,
                              env_config, 0,
                              eval_cfg.framework.eval_save_metrics,
                              eval_cfg.cinematic_recorder)

    del env_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(config_name='config', config_path='.')
def main(eval_cfg: DictConfig) -> None:
    logging.info('\n' + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = os.path.join(eval_cfg.framework.logdir,
                                eval_cfg.rlbench.task_name,
                                'RoboPrompt',
                                'seed%d' % start_seed)

    env_device = 'cuda'
    logging.info('Using env device %s.' % str(env_device))

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    eval_cfg.rlbench.cameras = CAMERAS
    
    obs_config = create_obs_config()      

    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    # single-task or multi-task
    if len(eval_cfg.rlbench.tasks) > 1:
        tasks = eval_cfg.rlbench.tasks
        multi_task = True

        task_classes = []
        for task in tasks:
            if task not in task_files:
                raise ValueError('Task %s not recognised!.' % task)
            task_classes.append(task_file_to_task_class(task))

        env_config = (task_classes,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      eval_cfg.framework.eval_episodes,
                      True,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)
    else:
        task = eval_cfg.rlbench.tasks[0]
        multi_task = False

        if task not in task_files:
            raise ValueError('Task %s not recognised!.' % task)
        task_class = task_file_to_task_class(task)

        env_config = (task_class,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      True,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)

    logging.info('Evaluating seed %d.' % start_seed)
    eval_seed(eval_cfg,
                logdir,
                eval_cfg.rlbench.cameras,
                env_device,
                multi_task, start_seed,
                env_config)

if __name__ == '__main__':
    main()
