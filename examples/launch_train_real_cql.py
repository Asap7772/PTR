import argparse
import sys
import imp
from jaxrl2.utils.general_utils import AttrDict

from examples.train_pixels_real import main
from jaxrl2.utils.launch_util import parse_training_args
import os
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,
                        help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=20000, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=64, help='Mini batch size.', type=int)
    parser.add_argument('--online_start', default=int(1e9), help='Number of training steps after which to start training online.', type=int)
    parser.add_argument('--max_steps', default=int(1e9), help='Number of training steps.', type=int)
    parser.add_argument('--tqdm', default=1, help='Use tqdm progress bar.', type=int)
    parser.add_argument('--save_video', action='store_true', help='Save videos during evaluation.')
    parser.add_argument('--use_negatives', action='store_true', help='Use negative_data')
    parser.add_argument('--reward_scale', default=11.0, help='Scale for the reward', type=float)
    parser.add_argument('--reward_shift', default=-1, help='Shift for the reward', type=float)
    parser.add_argument('--reward_type' ,default='final_one', help='reward type')

    parser.add_argument('--frame_stack', default=1, help='Number of frames stacked', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations',  type=int)
    parser.add_argument('--add_prev_actions', default=0, help='whether to add low-dim previous actions to the obervations', type=int)

    parser.add_argument('--dataset', default='online_reaching_pixels', help='name of dataset')
    parser.add_argument('--target_dataset', default='', help='name of dataset', type=str)

    parser.add_argument('--online_mixing_ratio', default=0.5,
                        help='fraction of batch composed of old data to be used, the remainder part is newly collected data',
                        type=int)
    parser.add_argument('--target_mixing_ratio', default=0.9,
                        help='fraction of batch composed of bridge data, the remainder is target data',
                        type=float)
    parser.add_argument("--multi_viewpoint", default=1, help="whether to use multiple camreas", type=int)
    parser.add_argument('--negatives_nofinal_bootstrap', action='store_true', help='apply bootstrapping at last time step of negatives')

    parser.add_argument('--trajwise_alternating', default=1,
                        help='alternate between training and data collection after each trajectory', type=int)
    parser.add_argument('--restore', action='store_true', help='whether to restore weights')
    parser.add_argument('--restore_path',
                        default='',
                        help='folder inside $EXP where weights are stored')
    parser.add_argument('--only_add_success', action='store_true', help='only add successful traj to buffer')

    parser.add_argument('--wandb_project', default='cql_real', help='wandb project')
    parser.add_argument('--wandb_user', default='frederik', help='wandb user config (this is not the actual wanddb username)')
    parser.add_argument("--use_terminals", default=1, help="whether to use terminals", type=int)

    parser.add_argument('--from_states', action='store_true', help='only use states, no images')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--online_from_scratch', action='store_true', help='train online from scratch.')

    parser.add_argument('--stochastic_data_collect', default=1, help='sample from stochastic policy for data collection.', type=int)

    parser.add_argument('--algorithm', default='cql_encodersep_parallel', help='type of algorithm')

    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--config', default='examples/configs/offline_pixels_default_real.py', help='File path to the training hyperparameter configuration.')
    parser.add_argument("--azure", action='store_true', help="run on azure")
    parser.add_argument("--offline_only", default=1, help="whether to only perform offline training", type=int)
    parser.add_argument("--eval_only", action='store_true', help="perform evals only")
    parser.add_argument('--rescale_actions', default=1, help='rescale actions to so that action-bounds are within +-1', type=int)
    parser.add_argument('--normalize_actions', default=0, help='rescale actions to so that action-bounds are within +-1', type=int)
    parser.add_argument('--start_transform', default='openmicrowave', help='start transform to use', type=str)
    
    #environment
    parser.add_argument('--episode_timelimit', default=40, help='prefix to use', type=int)

    parser.add_argument('--restore_reward_path', default='', help='File path to the weights file of the reward function inside $EXP folder')
    parser.add_argument('--file_system', default='', help='local or azure-blobfuse')

    parser.add_argument('--debug', action='store_true', help='abort saving to replay buffer early for testing   ')
    parser.add_argument('--num_eval_tasks', default=-1, help='nubmer of eval tasks, if -1 infer from dataset', type=int)
    parser.add_argument('--annotate_with_classifier', default=0,
                        help='whether to annotate bridge data with classifier rewads', type=int)

    parser.add_argument('--num_final_reward_steps', default=1, help='number of final reward timesteps', type=int)
    parser.add_argument('--term_as_rew', type=int, default=1)

    parser.add_argument('--offline_finetuning_start', default=-1, help='when to start offline finetuning', type=int)
    parser.add_argument('--autoregressive_launcher', default=0, help='when to start offline finetuning', type=int)
    
    parser.add_argument('--num_target_traj', default=-1,
                        help='num trajectories used for the target task, -1 means all',
                        type=int)

    # algorithm args:
    train_args_dict = dict(
        actor_lr= 1e-4,
        critic_lr= 3e-4,
        temp_lr= 3e-4,
        decay_steps= None,
        hidden_dims= (256, 256, 256),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 256,
        discount= 0.96,
        cql_alpha= 0.0,
        tau= 0.005,
        backup_entropy= False,
        target_entropy= None,
        critic_reduction= 'min',
        dropout_rate=0.0,
        init_temperature= 1.0,
        max_q_backup= True,
        policy_encoder_type='resnet_small',
        encoder_type='resnet_small',
        encoder_norm= 'batch',
        dr3_coefficient= 0.0,
        use_spatial_softmax=False,
        softmax_temperature=-1,
        use_spatial_learned_embeddings=True,
        share_encoders=False,
        use_bottleneck=True,
        use_action_sep=False,
        use_basis_projection=False,
        basis_projection_coefficient=0.0,
        use_multiplicative_cond=False,
        target_entropy_factor=1.0,
        use_normalized_features=False,
        policy_use_multiplicative_cond=False,
        use_pixel_sep=False,
        use_gaussian_policy=False,
        min_q_version=3,
        std_scale_for_gaussian_policy=0.05,
        q_dropout_rate=0.0,
        use_uds=False,
        use_cds=False,
        autoregressive_policy=0,
        autoregressive_repeat=0,
        autoregressive_project=0,
        autoregressive_qfunc=0,
        autoregressive_type=0,
        bc_start=0,
        bc_end=0,
    )

    variant, args = parse_training_args(train_args_dict, parser)

    filesys = 'AZURE_' if args.file_system == 'azure' else ''
    prefix = '/data/spt_data/experiments/' if args.azure else os.environ[filesys + 'EXP']
    if args.restore_reward_path:
        local_reward_restore_path = os.environ[filesys + 'EXP'] + '/' + args.restore_reward_path
        full_reward_restore_path = prefix + '/' + args.restore_reward_path
        config_file_reward = '/'.join(local_reward_restore_path.split('/')[:-1]) + '/config.json'
        with open(config_file_reward) as config_file:
            variant_reward = json.load(config_file)
        variant_reward = AttrDict(variant_reward)
        variant.variant_reward = variant_reward
    else:
        full_reward_restore_path = ''

    variant.restore_reward_path = full_reward_restore_path
    if variant.policy_encoder_type == 'same':
        variant.policy_encoder_type = variant.encoder_type
    
    if variant.use_gaussian_policy:
        variant.rescale_actions = 0
        variant.normalize_actions = 1

    if not args.azure:
        main(variant)
        sys.exit()
    else:
        from doodad.wrappers.easy_launch import sweep_function, save_doodad_config
        def train(doodad_config, default_params):
            main(variant)
            save_doodad_config(doodad_config)

        params_to_sweep = {}
        mode = 'azure'
        sweep_function(
            train,
            params_to_sweep,
            default_params={},
            log_path=args.prefix,
            mode=mode,
            use_gpu=True,
            num_gpu=1,
        )
