import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from agent.runners.DreamerRunner import DreamerRunner
from configs import Experiment # , SimpleObservationConfig, NearRewardConfig, DeadlockPunishmentConfig, RewardsComposerConfig
from configs.EnvConfigs import EnvCurriculumConfig, StarCraftConfig, PettingZooConfig, FootballConfig, MAMujocoConfig, SMACv2Config, MATEConfig


from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig

# for SMACv2
from configs.dreamer.smacv2.smacv2LearnerConfig import Smacv2DreamerLearnerConfig
from configs.dreamer.smacv2.smacv2ControllerConfig import Smacv2DreamerControllerConfig

# for MPE
from configs.dreamer.mpe.MpeLearnerConfig import MPEDreamerLearnerConfig
from configs.dreamer.mpe.MpeControllerConfig import MPEDreamerControllerConfig

# for GRF
from configs.dreamer.football.GRFLearnerConfig import GRFDreamerLearnerConfig
from configs.dreamer.football.GRFControllerConfig import GRFDreamerControllerConfig

# for MAMuJoCo
from configs.dreamer.mamujoco.mamujocoLearnerConfig import MAMujocoDreamerLearnerConfig
from configs.dreamer.mamujoco.mamujocoControllerConfig import MAMujocoDreamerControllerConfig

# for MATE
from configs.dreamer.mate.MATELearnerConfig import MATEDreamerLearnerConfig
from configs.dreamer.mate.MATEControllerConfig import MATEDreamerControllerConfig


from environments import Env
from utils import generate_group_name, format_numel_str_deci

import torch
import numpy as np
import random

from tb_logger import LOGGER

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="flatland", help='Flatland or SMAC env')
    parser.add_argument('--env_name', type=str, default="5_agents", help='Specific setting')
    parser.add_argument('--policy_class', type=str, required=True)

    # specialized arg for MAMujoco
    parser.add_argument('--agent_conf', type=str, default=None)
    # specialized arg for MPE
    parser.add_argument('--enable_mpe_disc', action='store_true')

    # specialized args for MATE
    parser.add_argument('--mate_levels', type=int, default=5, help='DiscreteCamera levels; action size is levels^2')
    parser.add_argument('--mate_episode_limit', type=int, default=200, help='Overrides max_episode_steps in the MATE scenario')

    parser.add_argument('--n_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--seed', type=int, default=1, help='Number of workers')
    parser.add_argument('--steps', type=int, default=1e6, help='Number of workers')
    parser.add_argument('--mode', type=str, default='disabled')
    parser.add_argument('--temperature', type=float, default=1.)  # for controller sampling data

    parser.add_argument('--sample_temp', type=float, default='inf')
    parser.add_argument('--ce_for_cont', action='store_true')
    parser.add_argument('--state_decoder_type', type=int, default=1)

    parser.add_argument('--load_pretrained', action='store_true', default=False)
    parser.add_argument('--load_path', type=str, default=None)

    # crash/interrupt resume: full model + optimizer + counters (unlike --load_path, which is eval-only)
    parser.add_argument('--resume_path', type=str, default=None,
                         help='Path to a latest_resume.pth checkpoint to continue an interrupted run from.')
    parser.add_argument('--save_interval', type=int, default=5000,
                         help='Env steps between lightweight eval-snapshot checkpoints.')
    parser.add_argument('--save_resume_every_episodes', type=int, default=1,
                         help='Episodes between full resume-checkpoint overwrites (latest_resume.pth).')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                         help='Fixed W&B run id so restarts on Kaggle append to the same run instead of forking a new one.')

    parser.add_argument('--use_tensorboard', action='store_true')

    return parser.parse_args()


def train_dreamer(exp, n_workers, resume_path=None, save_interval=5000, save_resume_every_episodes=1):
    runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers, resume_path=resume_path)
    runner.run(exp.steps, exp.episodes, save_interval=save_interval, save_mode='interval',
               resume_env_steps=runner.resume_env_steps, save_resume_every_episodes=save_resume_every_episodes)


def get_env_info(configs, env):
    if not env.discrete:
        assert hasattr(env, 'individual_action_space')
        individual_action_space = env.individual_action_space
    else:
        individual_action_space = None

    for config in configs:
        config.IN_DIM = env.n_obs
        config.STATE_DIM = env.state_dim
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
        config.CONTINUOUS_ACTION = not env.discrete
        config.ACTION_SPACE = individual_action_space

        ## debug in SMAC
        # config.nf_al = env.nf_al
        # config.nf_en = env.nf_en

        # config.n_allies  = env.n_allies
        # config.n_enemies = env.n_enemies
        
    
    print(f'Observation dims: {env.n_obs}')
    print(f'Global State dims: {env.state_dim}')
    print(f'Action dims: {env.n_actions}')
    print(f'Num agents: {env.n_agents}')
    print(f'Continuous action for control? -> {not env.discrete}')
    
    if hasattr(env, 'individual_action_space'):
        print(f'Individual action space: {env.individual_action_space}')

    env.close()


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 2000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_smacv2_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = SMACv2Config(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 2000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_pettingzoo_configs(env_name, continuous_action = True):
    agent_configs = [MPEDreamerControllerConfig(), MPEDreamerLearnerConfig()]
    env_config = PettingZooConfig(env_name, RANDOM_SEED, continuous_action)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_football_configs(env_name):
    agent_configs = [GRFDreamerControllerConfig(), GRFDreamerLearnerConfig()]
    env_config = FootballConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_mate_configs(env_name, levels, episode_limit):
    agent_configs = [MATEDreamerControllerConfig(), MATEDreamerLearnerConfig()]
    env_config = MATEConfig(env_name, RANDOM_SEED, levels=levels, max_episode_steps=episode_limit)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_mamujoco_configs(scenario, agent_config):
    agent_configs = [MAMujocoDreamerControllerConfig(), MAMujocoDreamerLearnerConfig()]
    env_config = MAMujocoConfig(scenario = scenario, seed = RANDOM_SEED, agent_conf = agent_config)

    agent_configs[1].env_name = scenario

    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    RANDOM_SEED = 23
    args = parse_args()
    RANDOM_SEED += args.seed * 100
    if args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.SMACv2:
        configs = prepare_smacv2_configs(args.env_name)
    elif args.env == Env.PETTINGZOO:
        configs = prepare_pettingzoo_configs(args.env_name, continuous_action=not args.enable_mpe_disc) # continuous_action=True)
    elif args.env == Env.GRF:
        configs = prepare_football_configs(args.env_name)
    elif args.env == Env.MAMUJOCO:
        configs = prepare_mamujoco_configs(args.env_name, args.agent_conf)
    elif args.env == Env.MATE:
        configs = prepare_mate_configs(args.env_name, args.mate_levels, args.mate_episode_limit)
    else:
        raise Exception("Unknown environment")
    
    # seed everywhere
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    torch.autograd.set_detect_anomaly(True)
    # --------------------

    assert args.state_decoder_type in [1, 2]
    
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)
    configs["learner_config"].seed = RANDOM_SEED

    configs["learner_config"].policy_class    = args.policy_class
    configs["controller_config"].policy_class = args.policy_class

    if args.policy_class == 'gaussian':
        configs['learner_config'].ENTROPY = 0.001
    elif args.policy_class == 'beta':
        configs['learner_config'].ENTROPY = 0.01

    # param overwrite
    configs["learner_config"].use_ce_for_cont   = args.ce_for_cont
    configs['learner_config'].compute_end_in_TD = args.ce_for_cont      # When using CE for cont prediction, we would compute return with binary termination
    configs["learner_config"].diffusion_sampler_cfg.num_steps_denoising = configs["learner_config"].NUM_AGENTS if configs["learner_config"].NUM_AGENTS > 2 \
        else configs["learner_config"].NUM_AGENTS * 2

    rewards_prediction_config = getattr(configs["learner_config"], 'rewards_prediction_config', None)

    configs["learner_config"].load_pretrained = args.load_pretrained
    configs["learner_config"].load_path = args.load_path

    if args.sample_temp == float('inf'):
        configs["learner_config"].sample_temperature = str(args.sample_temp)
    else:
        configs["learner_config"].sample_temperature = args.sample_temp

    ## newly added
    configs["learner_config"].state_decoder_type = "s + id" if args.state_decoder_type == 1 else "s + last_obs"
    configs["controller_config"].state_decoder_type = "s + id" if args.state_decoder_type == 1 else "s + last_obs"

    current_date = datetime.now()
    current_date_string = current_date.strftime("%m%d")

    # make run directory
    dir_prefix = args.env_name + '-'+ args.agent_conf if args.agent_conf is not None else args.env_name

    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f"/{current_date_string}_results") / args.env / (dir_prefix)
    # curr_run = f"run{random.randint(1000, 9999)}"
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        os.makedirs(str(run_dir / "ckpt"))

    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "agent"), dst=run_dir / "agent")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "configs"), dst=run_dir / "configs")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "networks"), dst=run_dir / "networks")
    shutil.copyfile(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "train.py"), dst=run_dir / "train.py")
    
    print(f"Run files are saved at {str(run_dir)}\n")
    # -------------------

    configs["learner_config"].RUN_DIR = str(run_dir)
    configs["learner_config"].map_name = args.env_name

    if args.env == Env.MAMUJOCO:
        group_name = f"raw_trans_branch_{args.env_name}_{args.agent_conf}_H{configs['learner_config'].horizon}"

    else:
        group_name = f"raw_trans_branch_{args.env_name}_H{configs['learner_config'].horizon}"

    if args.ce_for_cont:
        group_name += f"_ce_for_cont"

    ## t -> policy sample temperature; s -> seed; i -> sample interval; H -> imagination horizon
    ## o1 -> transformer based pcont NLL prediction, end threshold is 0.7
    if args.env == Env.MAMUJOCO:
        run_name = f"({current_date_string}) raw_H{configs['learner_config'].horizon}_s{RANDOM_SEED}_i{configs['learner_config'].N_SAMPLES}_{args.policy_class}_gamma{configs['learner_config'].GAMMA}"
    else:
        run_name = f"({current_date_string}) raw_H{configs['learner_config'].horizon}_t{args.temperature}_s{RANDOM_SEED}_i{configs['learner_config'].N_SAMPLES}_{args.policy_class}_gamma{configs['learner_config'].GAMMA}"

    run_name += f"_DecObs_original"
    run_name += f"_{configs['learner_config'].vq_type}"
    # run_name += f"_o4"

    # EC denotes entropy coefficient
    job_type = f"{args.policy_class}_gamma{configs['learner_config'].GAMMA}_EC{configs['learner_config'].ENTROPY}"
    if configs['learner_config'].critic_dist_config['loss_type'] != 'regression':
        job_type += f"_{configs['learner_config'].critic_dist_config['loss_type']}{configs['learner_config'].critic_dist_config['bins']}"

    else:
        job_type += f"_{configs['learner_config'].critic_dist_config['loss_type']}_Tau{configs['learner_config'].tau}"
    
    # DN denotes Denoiser max grad norm
    # job_type += f"_DN{configs['learner_config'].denoiser_max_grad_norm}"

    # w/o_VN denotes no usage of value normalization
    if configs['learner_config'].use_valuenorm:
        job_type += f"_w/VN"
    
    else:
        job_type += f"_w/o_VN"

    if configs['learner_config'].compute_end_in_TD:
        job_type += f"_w/end"
    else:
        job_type += f"_w/o_end"

    global wandb
    import wandb
    wandb.init(
        project='SMAD' if args.env != Env.MAMUJOCO else 'mamujoco',
        mode=args.mode if not args.load_pretrained else 'disabled',
        group=group_name,
        job_type=job_type,
        name=run_name,
        config=configs["learner_config"].to_dict(),
        notes="",
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id is not None else None,
    )

    print("group name: ", group_name)
    print("run name: ", run_name)
    print("job type: ", job_type)

    ## tensorboard initialize
    exp_dir = 'tb_logs/' + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{group_name}_s{RANDOM_SEED}_i{configs['learner_config'].N_SAMPLES}_{args.policy_class}_gamma{configs['learner_config'].GAMMA}"
    if args.use_tensorboard:
        LOGGER.initialize(log_dir=exp_dir)

    exp = Experiment(steps=args.steps,
                     episodes=500000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])

    train_dreamer(exp, n_workers=args.n_workers, resume_path=args.resume_path,
                  save_interval=args.save_interval, save_resume_every_episodes=args.save_resume_every_episodes)
