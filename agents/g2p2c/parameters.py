def set_args(args):
    #args.action_type = 'exponential'  # 'normal', quadratic, proportional_quadratic, exponential
    args.feature_history = 12 # Number of past observations the model uses for context
    args.calibration = 12
    args.action_scale = 5 # Scales actions to limit their range
    args.insulin_max = 5 # Sets a cap on the insulin action's upper limit
    args.n_features = 2 # Number of featuargs.n_features = 2 # Number of features used in each time stepres used in each time step
    args.n_activity_features = 2
    args.t_meal = 20 # Time delay for meals in minutes

    # Annoucement features
    args.use_meal_announcement = False  # adds meal announcement as a timeseries feature.
    args.use_carb_announcement = False # Whether carbohydrate information is included as a feature.
    args.use_tod_announcement = False # Includes time-of-day data

    # Handcrafted features
    args.use_handcraft = 0 # Defines if handcrafted features are used.
    args.n_handcrafted_features = 1

    # Model architecture settings
    args.n_hidden = 16 
    args.n_rnn_layers = 1 # Number of RNN layers for time-series data.
    args.rnn_directions = 1  # Sets if the RNN is unidirectional (1) or bidirectional (2).
    args.bidirectional = False

    # Episode and step configuration
    args.max_epi_length = 288 * 10
    args.n_step = 256 # Step count per batch for policy and value function updates.
    args.max_test_epi_len = 288 

    # Reward and return settings
    #args.return_type = 'average'   # discount | average
    args.gamma = 1 if args.return_type == 'average' else 0.99
    args.lambda_ = 1 if args.return_type == 'average' else 0.95

    # Regularization parameters
    args.entropy_coef = 0.001
    args.grad_clip = 20
    args.eps_clip = 0.1
    args.target_kl = 0.01 # Target KL divergence for policy updates, balancing between old and new policies.
    args.normalize_reward = True # Normalizes rewards across episodes to stabilize training.

    # Rollout and worker setup
    args.shuffle_rollout = True
    args.n_training_workers = 16 if args.debug == 0 else 2
    args.n_testing_workers = 20 if args.debug == 0 else 2

    # Epoch and learning rate settings for policy and value function
    args.n_pi_epochs = 5
    args.n_vf_epochs = 5
    args.pi_lr = 1e-4 * 3
    args.vf_lr = 1e-4 * 3
    args.batch_size = 1024

    # aux model learning
    args.aux_buffer_max = 25000 if args.debug == 0 else 1000  # must be larger than steps at one update
    args.aux_frequency = 1  # frequency of updates
    args.aux_vf_coef = 0.01
    args.aux_pi_coef = 0.01
    args.aux_batch_size = 1024
    args.n_aux_epochs = 5
    args.aux_lr = 1e-4 * 3

    # plannning
    args.planning_n_step = 6
    args.n_planning_simulations = 50
    args.plan_batch_size = 1024
    args.n_plan_epochs = 1
    #args.planning_lr = 1e-4 * 3

    return args