
import copy

configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        batch_size=32,
        learning_rate=1.5e-4,  # Adam Optimizer
        target_update_interval=256,
        clip_grad_norm=40.,
        min_steps_rl=int(5e3),
        double_dqn=False,
        prioritized_replay=False,
        n_step_return=1,
        replay_size=int(1e5),
        min_steps_ul=0,
        max_steps_ul=None,
        ul_learning_rate=1e-3,
        ul_optim_kwargs=None,
        # ul_replay_size=1e5,
        ul_update_schedule="constant_1",
        ul_lr_schedule="cosine",
        ul_lr_warmup=0,
        ul_delta_T=3,
        ul_batch_B=32,
        ul_batch_T=16,
        ul_random_shift_prob=0.1,
        ul_random_shift_pad=4,
        ul_target_update_interval=1,
        ul_target_update_tau=0.01,
        ul_latent_size=256,
        ul_anchor_hidden_sizes=512,
        ul_clip_grad_norm=10.,
        # ul_pri_alpha=0.,
        # ul_pri_beta=1.,
        # ul_pri_n_step_return=1,
    ),
    env=dict(
        seed=0,
        goal_id=1
    ),
    # Will use same args for eval env.
    model=dict(
        channels=[32, 16],
        kernel_sizes=[4, 4],
        strides=[1, 2],
        # dueling=False,
        paddings=[1, 2],  # No padding for standard 84x84 images.
        stop_conv_grad=False,
        hidden_sizes=None,
        kiaming_init=True,
    ), 
    optim=dict(eps=0.01 / 256),
    runner=dict(
        n_steps=3200000,
        log_interval_steps=320000,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=32,
        max_decorrelation_steps=1000,
        eval_n_envs=4,
        eval_max_steps=int(150e3),
        eval_max_trajectories=75,
    ),
) 

configs["scaled_ddqn_ul"] = config

