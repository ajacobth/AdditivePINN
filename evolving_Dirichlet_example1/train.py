

#y (bead width direction)
#^
#|
#|
#|
#|-------------->x deposition direction
#0

import os
import time
import shutil 

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections
from absl import logging
import wandb

from A3DPINN.samplers import UniformSampler, SeqBaseSampler, SeqBoundarySampler, SeqInitialSampler, StepIndexSampler
from A3DPINN.logging import Logger
from A3DPINN.utils import save_checkpoint
import orbax 
import models
from utils import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    

    #t0 = config.dimensions.t_min
    #t1 = config.dimensions.t_max
    
    #x0 = config.dimensions.x_min
    #x1 = config.dimensions.x_max
    
    #y0 = config.dimensions.y_min # bead dimension
    #y1 = config.dimensions.y_max # beaed dimnesion
    
    
    # Define domain
    #dom = jnp.array([[t0, t1], [x0, x1]])

    # Define residual sampler
    time_sampler = iter(UniformSampler(jnp.array([[0.0, 1.]]), config.training.time_batch_size_per_device))
    

    
    
    init_len = (config.process_conditions.init_length)/config.dimensions.x_max
    initial_sampler = iter(UniformSampler(jnp.array([[0., 1.],[0., 0.], [0., 1.0]]), config.training.batch_size_per_device))
    
    step_sampler= iter(StepIndexSampler(100000, config.training.time_batch_size_per_device))

    # Initialize model
    model = models.A3DHeatTransfer(config)


    path = os.path.join(workdir, "ckpt", config.wandb.name)
    if os.path.exists(path):
        shutil.rmtree(path)
        
    # Initialize evaluator
    evaluator = models.A3DHeatTransferEvaluator(config, model)
    
    mgr_options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=3)
    ckpt_mgr = orbax.checkpoint.CheckpointManager(path, orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)
    print("Waiting for JIT...")
    start_time_total = time.time()
    for step in range(config.training.max_steps):
        start_time = time.time()

        time_batch = next(time_sampler)
        step_batch = next(step_sampler)
        batch_initial = next(initial_sampler)

        model.state = model.step(model.state, time_batch, batch_initial,  step_batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state,time_batch,
                                                   batch_initial, step_batch)


        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                time_batch= jax.device_get(tree_map(lambda x: x[0], time_batch))
                batch_initial= jax.device_get(tree_map(lambda x: x[0], batch_initial))
                step_batch = jax.device_get(tree_map(lambda x: x[0], step_batch))
                
                log_dict = evaluator(state, time_batch, batch_initial, step_batch)
                wandb.log(log_dict, step)

                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                
                save_checkpoint(model.state, path, ckpt_mgr)

    f=open("time_summary.txt", "a")
    f.write("\n"+ config.wandb.name+"--- %s seconds ---" % (time.time() - start_time_total))
    f.close()

    return model
