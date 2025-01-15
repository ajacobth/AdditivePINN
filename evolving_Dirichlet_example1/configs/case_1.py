import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "LowKxx_2D_Additive_paper_BC_Bottom_topNBC_conv"
    wandb.name = "case1_MLP"
    wandb.tag = None
    
    # material properties
    config.material_properties = material_properties = ml_collections.ConfigDict()
    material_properties.density = 1.
    material_properties.specific_heat = 100.
    material_properties.thermal_conductivity_xx = 1.
    material_properties.thermal_conductivity_yy = 6.
    material_properties.heat_transfer_coefficient = 15.
    
    # processing conditions
    config.process_conditions = process_conditions = ml_collections.ConfigDict()
    process_conditions.deposition_temperature = 2.
    process_conditions.bed_temperature = 0.5
    process_conditions.print_speed = 1. # currently only in x-direction.
    process_conditions.velocity_vector = jnp.array([1.0, 0.])
    process_conditions.init_length = jnp.array([0.01])
    process_conditions.bead_width = 1.0
    
    # ambient conditions
    process_conditions.ambient_convection_temp = 0.
    process_conditions.ambient_radiation_temp = 0. # will calibrate later

    # dimensions
    config.dimensions = dimensions = ml_collections.ConfigDict()
    dimensions.t_min = 0.0
    dimensions.x_min = 0.0
    dimensions.y_min = 0.0
    dimensions.t_max = 2.5
    
    # define bead dimensions below
    dimensions.x_max = 2.51  # bead length
    dimensions.y_max = 1.0  # bead with
    
    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 4
    arch.hidden_dim = 64
    arch.out_dim = 1
    arch.activation = "tanh"
    #arch.periodicity = False#ml_collections.ConfigDict(
        #{"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    #)
    #arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 128})
    #arch.reparam = ml_collections.ConfigDict(
    #   {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    #)
    
    
    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 200000
    training.batch_size_per_device = 2048
    training.time_batch_size_per_device = 2
    training.test_batch_size_per_device = 32
    training.loss_type = "strong"
    
    # Test Functions - if using weak type loss
    config.test_functions = test_functions = ml_collections.ConfigDict()
    test_functions.n_test = 200
    test_functions.lengthscale = 5e-3
    test_functions.centers_type = 'equidistant'
    
    
    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 5e-4
    optim.decay_rate = 0.9
    optim.decay_steps = 5000
    optim.grad_accum_steps = 0


    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({ "wall":1., "dbc_b1":1., "nbc_b2":1.,"evol_init":1.,"res": 1.})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = False
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = False
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = True
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 101

    return config
