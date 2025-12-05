class MPOConfig:
    def __init__(
        self,
        env_name="HalfCheetah-v5",
        batch_size=256,
        num_training_episodes=1000,
        num_candidate_actions=32,
        min_replay_size=30_000,
        num_optimization_steps_per_step=2,
        q_lr=0.3e-4,
        pi_lr=0.3e-4,
        tau=0.005,
        dual_lr=1e-3,
        eta=1.0,
        kl_epsilon=0.2,
        policy_old_sync_frequency=50,
        log_dir="./logs/mpo_experiment",
        eval_freq=10,
        eval_episodes=5,
        seed=42,
        entropy_coeff=1e-3,
        checkpoint_ep_freq=50,
        e_step_solve_dual=False,
        *args,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.num_training_episodes = num_training_episodes
        self.num_candidate_actions = num_candidate_actions
        self.min_replay_size = min_replay_size
        self.num_optimization_steps_per_step = num_optimization_steps_per_step
        self.q_lr = q_lr
        self.pi_lr = pi_lr
        self.tau = tau
        self.dual_lr = dual_lr
        self.eta = eta
        self.kl_epsilon = kl_epsilon
        self.policy_old_sync_frequency = policy_old_sync_frequency
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.entropy_coeff = entropy_coeff
        self.env_name = env_name
        self.checkpoint_ep_freq = checkpoint_ep_freq
