#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:10:00 2024

@author: akshayjacobthomas
"""

import os

import ml_collections
from jax import vmap
import jax.numpy as jnp

import matplotlib.pyplot as plt

from A3DPINN.utils import restore_checkpoint

import models
from utils import get_dataset
from get_solution import get_solution, get_solution_plot
import glob
from PIL import Image
import numpy as np
import matplotlib.animation as animation
from matplotlib import ticker

import contextlib
plt.rcParams['text.usetex'] = True

    
def create_gif(folder_path, gif_name, size=(300, 300)):
    images = []
    for filename in sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(folder_path, filename))
            #img = img.resize(size, Image.ANTIALIAS)
            images.append(img)

    # Save as GIF
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=100, loop=0)

def plot_point_history(config: ml_collections.ConfigDict, workdir,
                       model, params,
                       xy_loc, 
                       node_x,
                       node_y,
                       time_temp_history, name:str):
    
    u_max = config.process_conditions.deposition_temperature
    t_max = config.dimensions.t_max
    x_max = config.dimensions.x_max
    y_max = config.dimensions.y_max
    
    print_speed = config.process_conditions.print_speed
    
    # compare time tempearture histpry for  few points
    
    act_time = xy_loc[0]/print_speed
    times_post_activation = jnp.linspace(act_time, t_max, 100)
    temp_pred_post_activation = model.evalfn_(params, times_post_activation/t_max, xy_loc[0]/x_max, xy_loc[1]/y_max)*u_max
    
    # append deposition temperature 
    times_pre_activation = jnp.linspace(0.,act_time, 20)
    temp_pred_pre_activation = u_max*jnp.ones(20)
    
    time_array = jnp.concatenate((times_pre_activation, times_post_activation))
    temp_pred = jnp.concatenate((temp_pred_pre_activation, temp_pred_post_activation))
    # load FE results
    
    index_x = np.argwhere(np.abs(node_x-xy_loc[0])<1e-5)
    index_y = np.argwhere(np.abs(node_y-xy_loc[1])<1e-5)
    point_loc = np.intersect1d(index_x, index_y)
    times_FE = jnp.linspace(0., 2.5, int(2.5/0.01))              
    temp_FE = time_temp_history[:, point_loc]    
    # plot
    fig = plt.figure(figsize=(3, 2))
    #:plt.subplot(1, 1)
    
    plt.plot(times_FE, temp_FE, '*', color='k', markersize=2.0, label = "FE Prediction")
    plt.plot(time_array, temp_pred, color='r',linestyle='--', linewidth=1.5,label="PINN Prediction")
    plt.ylim([0., 2.5])
    plt.xlim([0., 2.5])
    plt.xlabel("$t$", fontsize=12)
    plt.ylabel("$u$", fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.axvline(x = act_time, color = 'k', linestyle = '--', label='Activation time')
    #plt.legend(fontsize=10)
    plt.title("x=%1.1f" %xy_loc[0] + ", y=%1.1f"%xy_loc[1])
    plt.tight_layout()
    
    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, name)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)

def intersect_ES(time:float, 
                 init_length:float, 
                 speed_x:float, 
                 velocity_vector:np.array,
                node_x:np.array,
                node_y:np.array):
    """time: time at which the nodes need to be exstracted
    init_length: amount of material already deposited at time=0.
    speed_x = printing speed"""
    
    x_length  = time*speed_x*velocity_vector[0] + init_length
    node_locations_x = np.argwhere(node_x<x_length).flatten()
    node_locations_y = np.argwhere(node_y<1.0).flatten()
    node_locations = np.intersect1d(node_locations_x, node_locations_y)
    print(node_locations.shape)
    return node_x[node_locations], node_y[node_locations], node_locations

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    
    # Restore model
    model = models.A3DHeatTransfer(config)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    state = restore_checkpoint(model.state, ckpt_path)
    #model.state = state['params']
    params = state['params']
    
    u_max = config.process_conditions.deposition_temperature
    t_max = config.dimensions.t_max
    x_max = config.dimensions.x_max
    y_max = config.dimensions.y_max
    
    # load FE data
    node_coords = np.genfromtxt('NodeCoords.csv', delimiter=',')
    #node_labels = np.genfromtxt('NodeLabels.csv', delimiter=',')
    node_x = node_coords[:, 0]
    node_y = node_coords[:, 1]
    #node_z = node_coords[:, 2]
    time_temp_history = np.genfromtxt('time_history_output.csv', delimiter=',') # shape is time_incrementsXnumber of nodes
    
    
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    xy = jnp.array([0.0, 0.0])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=0.0,y=0.0.pdf")
    
    xy = jnp.array([0.0, 0.1])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=0.0,y=0.1.pdf")
    
    xy = jnp.array([0.0, 0.4])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=0.0,y=0.4.pdf")
    
    xy = jnp.array([0.0, 0.75])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=0.0,y=0.75.pdf")
    
    

    xy = jnp.array([1.0, 0.0])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=1.0,y=0.0.pdf")
    
    xy = jnp.array([1.0, 0.1])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=1.0,y=0.1.pdf")
    
    xy = jnp.array([1.0, 0.4])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=1.0,y=0.4.pdf")
    
    xy = jnp.array([1.0, 0.75])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=1.0,y=0.75.pdf")
    
    
 
    xy = jnp.array([2.0, 0.])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=2.0,y=0.25.pdf")
    
    xy = jnp.array([2.0, 0.1])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=2.0,y=0.1.pdf")
    
    xy = jnp.array([2.0, 0.4])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=2.0,y=0.4.pdf")
    
    xy = jnp.array([2.0, 0.75])
    # plot point history
    plot_point_history(config, workdir, model, params, xy, node_x, node_y, time_temp_history, "At_x=2.0,y=0.75.pdf")
    
    

    # STILL EVOLUTION IMAGES - ----- - -   - - ------
    
    save_evol_dir = os.path.join(save_dir, "evolution")
    
    if not os.path.isdir(save_evol_dir):
        os.makedirs(save_evol_dir)
    time_array = jnp.array([0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])#jnp.linspace(0., 2.0, 5)
    
    time_array_scaled = time_array/t_max
    
    fig, axs = plt.subplots(time_array.shape[0])
    for i in range(time_array.shape[0]):
        u_pred, x_, y_ = model.evaluate_Uplot(params, time_array_scaled[i])
        
        
        sc = axs[i].scatter(x_, y_, s=4, c=u_pred, vmin = 0., vmax = 2.0, cmap=plt.cm.get_cmap('jet'))
        
        axs[i].set_xlim([0.,2.5])
        axs[i].set_ylim([0.,1.])
        #axs[i].set_xlabel('$x$', fontsize=18)
        #axs[i].set_ylabel('$y$', fontsize=18)
        plt.colorbar(sc, ax=axs[i])
        cbar = plt.colorbar(sc, ax=axs[i])
        cbar.ax.set_ylabel('$u$', fontsize=22)
        cbar.ax.tick_params(labelsize=22)
        axs[i].set_xlim([0., 2.5])
        axs[i].set_title('FE solution', fontsize=24)
        axs[i].set_ylim([0., 1.0])
        axs[i].set_xlabel('$x$', fontsize=22)
        axs[i].set_ylabel('$y$', fontsize=22)
        axs[i].tick_params(labelsize=22)

        
        
    fig_path = os.path.join(save_evol_dir, "ac.pdf")
    fig.savefig(fig_path, dpi=300)

        


    # plot initial condition everywhere
    
    fig, axs = plt.subplots(1,1)
    u_pred, x_, y_ = model.evaluate_init_plot(params)

    
            
    sc = axs.scatter(x_, y_, s=4, c=u_pred, vmin = 0., vmax = 2.5, cmap=plt.cm.get_cmap('jet'))
    
    axs.set_xlim([0.,2.5])
    axs.set_ylim([0.,1.])

        
        
    fig_path = os.path.join(save_evol_dir, "init.pdf")
    fig.savefig(fig_path, dpi=300)
    
    
    # EVOLUTION IMAGES GIF 
    
    save_evol_dir_gif= os.path.join(save_dir, "Pictures")
    
    if not os.path.isdir(save_evol_dir_gif):
        os.makedirs(save_evol_dir_gif)
    
    time_array = jnp.linspace(0.,2.5, 100)
    
    time_array_scaled = time_array/t_max
        
        
    for i in range(time_array.shape[0]):
        fig, axs = plt.subplots(1,1, figsize = (12, 3))
        u_pred, x_, y_ = model.evaluate_Uplot(params, time_array_scaled[i])
        
        
        sc = axs.scatter(x_, y_, s=4, c=u_pred, vmin = 0., vmax = 2.0, cmap=plt.cm.get_cmap('jet'))
        
        axs.set_xlim([0.,2.5])
        axs.set_ylim([0.,1.])
        axs.set_title(f"time={time_array[i]:.2f}s")
        axs.set_xlabel('$x$', fontsize=14)
        axs.set_ylabel('$y$', fontsize=14)
        plt.colorbar(sc, ax=axs)
        
        fig_path = os.path.join(save_evol_dir_gif, f"pic_{i+1}.png")
        fig.savefig(fig_path, dpi=300)
        plt.close()
        
    # create     
    directory = os.path.join(save_evol_dir_gif, "*.png")
    gif_path = os.path.join(save_evol_dir, "animation_gif.gif")
    
    #create_gif(directory, 'animation_true.gif')
    
    with contextlib.ExitStack() as stack:

        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(directory)))
    
        # extract  first image from iterator
        img = next(imgs)
    
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=gif_path, format='GIF', append_images=imgs,
        save_all=True, duration=15, loop=0)

# -------------------------- compare to FE ------------------------------

# ------------------------- also estimate L2 error -------------
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    save_FE_dir= os.path.join(save_dir, "FE_comparison")
    
    if not os.path.isdir(save_FE_dir):
        os.makedirs(save_FE_dir)
        

                             
    
    
    time_increment_abq=1e-2
    #element_size = 0.05
    #print_speed = 0.5
    nugget=0.001 # since we activate the first chunk in one go
    temp_history_locs = jnp.arange(0, int((t_max+0.1)/time_increment_abq), 10)
    time_stamps = temp_history_locs * time_increment_abq
    time_array_scaled = time_stamps/t_max
    l2_error_list = np.zeros(time_stamps.shape[0])

    for i in range(time_stamps.shape[0]-1):
        temp_history_loc = int(temp_history_locs[i])
        x, y, locs= intersect_ES(time_stamps[i], 0., 1., np.array([1., 0.]), node_x, node_y)
        FE_solution = time_temp_history[temp_history_loc, locs]
        x_jnp = jnp.asarray(x)/model.x_max
        y_jnp = jnp.asarray(y)/model.y_max
        PINN_solution = model.u_pred_fn(params, time_array_scaled[i], x_jnp, y_jnp)*u_max
        MAE = jnp.abs(FE_solution-PINN_solution)
        
        l2_error = jnp.linalg.norm(MAE)/jnp.linalg.norm(FE_solution)
        l2_error_list[i]=l2_error
        xx, yy = x, y
        
        fig, ax = plt.subplots(1,3, figsize=(26,5))
        sc = ax[0].scatter(xx, yy, s=2, c=FE_solution, vmin = 0., vmax = 2.0, cmap=plt.cm.get_cmap('jet'))
        cbar = plt.colorbar(sc, ax=ax[0])
        #cbar.ax.set_ylabel('$u$', fontsize=30)
        cbar.ax.tick_params(labelsize=26)
        ax[0].set_xlim([0., 2.5])
        ax[0].set_title('FE solution', fontsize=30)
        ax[0].set_ylim([0., 1.0])
        ax[0].set_xlabel('$x$', fontsize=26)
        ax[0].set_ylabel('$y$', fontsize=26)
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[0].tick_params(labelsize=26)
        
        sc = ax[1].scatter(xx, yy, s=2, c=PINN_solution, vmin = 0., vmax = 2.0, cmap=plt.cm.get_cmap('jet'))
        cbar = plt.colorbar(sc, ax=ax[1])
        #cbar.ax.set_ylabel('$u$', fontsize=30)
        cbar.ax.tick_params(labelsize=26) 
        ax[1].set_xlim([0., 2.5])
        ax[1].set_title('PINN solution', fontsize=30)
        ax[1].set_ylim([0., 1.0])
        ax[1].set_xlabel('$x$', fontsize=26)
        ax[1].set_ylabel('$y$', fontsize=26)
        ax[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[1].tick_params(labelsize=26)
        
        sc = ax[2].scatter(xx, yy, s=2, c=MAE, vmin = 0., vmax = 0.2, cmap=plt.cm.get_cmap('jet'))
        cbar = plt.colorbar(sc, ax=ax[2])
        #cbar.ax.set_ylabel('$u$', fontsize=30)
        cbar.ax.tick_params(labelsize=26)
        ax[2].set_xlim([0., 2.5])
        ax[2].set_title('Absolute error', fontsize=30)
        ax[2].set_ylim([0., 1.0])
        ax[2].set_xlabel('$x$', fontsize=26)
        ax[2].set_ylabel('$y$', fontsize=26)
        ax[2].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[2].tick_params(labelsize=26)
        
        fig_path = os.path.join(save_FE_dir, f"pic_{i+1}.png")
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    # plot the last step 
    temp_history_loc = 248
    x, y, locs= intersect_ES(2.49, 0., 1., np.array([1., 0.]), node_x, node_y)
    FE_solution = time_temp_history[temp_history_loc, locs]
    x_jnp = jnp.asarray(x)/model.x_max
    y_jnp = jnp.asarray(y)/model.y_max
    PINN_solution = model.u_pred_fn(params, 2.49/t_max, x_jnp, y_jnp)*u_max
    MAE = jnp.abs(FE_solution-PINN_solution)
    
    l2_error = jnp.linalg.norm(MAE)/jnp.linalg.norm(FE_solution)
    l2_error_list[-1]=l2_error
    xx, yy = x, y
    
    fig, ax = plt.subplots(1,3, figsize=(26,5))
    sc = ax[0].scatter(xx, yy, s=2, c=FE_solution, vmin = 0., vmax = 2.0, cmap=plt.cm.get_cmap('jet'))
    cbar = plt.colorbar(sc, ax=ax[0])
    #cbar.ax.set_ylabel('$u$', fontsize=30)
    cbar.ax.tick_params(labelsize=26)
    ax[0].set_xlim([0., 2.5])
    ax[0].set_title('FE solution', fontsize=30)
    ax[0].set_ylim([0., 1.0])
    ax[0].set_xlabel('$x$', fontsize=26)
    ax[0].set_ylabel('$y$', fontsize=26)
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[0].tick_params(labelsize=26)
    
    sc = ax[1].scatter(xx, yy, s=2, c=PINN_solution, vmin = 0., vmax = 2.0, cmap=plt.cm.get_cmap('jet'))
    cbar = plt.colorbar(sc, ax=ax[1])
    #cbar.ax.set_ylabel('$u$', fontsize=30)
    cbar.ax.tick_params(labelsize=26) 
    ax[1].set_xlim([0., 2.5])
    ax[1].set_title('PINN solution', fontsize=30)
    ax[1].set_ylim([0., 1.0])
    ax[1].set_xlabel('$x$', fontsize=26)
    ax[1].set_ylabel('$y$', fontsize=26)
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1].tick_params(labelsize=26)
    
    sc = ax[2].scatter(xx, yy, s=2, c=MAE, vmin = 0., vmax = 0.2, cmap=plt.cm.get_cmap('jet'))
    cbar = plt.colorbar(sc, ax=ax[2])
    #cbar.ax.set_ylabel('$u$', fontsize=30)
    cbar.ax.tick_params(labelsize=26)
    ax[2].set_xlim([0., 2.5])
    ax[2].set_title('Absolute error', fontsize=30)
    ax[2].set_ylim([0., 1.0])
    ax[2].set_xlabel('$x$', fontsize=26)
    ax[2].set_ylabel('$y$', fontsize=26)
    ax[2].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[2].tick_params(labelsize=26)
    
    fig_path = os.path.join(save_FE_dir, f"pic_last.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    
    # plot the l2 error
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    ax.plot(time_stamps, l2_error_list, '*', color ='k',markersize=6)
    ax.set_title('Rel. $L_2$ error', fontsize = 32)
    ax.set_ylim([0., 0.05])
    l2_path = os.path.join(save_FE_dir, "L2_error.csv")
    np.savetxt(l2_path, l2_error_list, delimiter=",")
    
    
    # change the fontsize
    ax.set_xlabel('$t$', fontsize=26)
    ax.set_ylabel('Rel. $L_2$ error', color='k',fontsize=26)
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    name = config.wandb.name + "Rel_l2.png"
    fig_path = os.path.join(save_dir, name)
    ax.tick_params(labelsize=26)
    fig.savefig(fig_path, dpi=300)
    
    l2_path = os.path.join(save_dir, "L2_error.csv")
    np.savetxt(l2_path, l2_error_list, delimiter=",")
    
    
