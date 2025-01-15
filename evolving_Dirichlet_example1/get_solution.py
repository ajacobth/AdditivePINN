#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:32:52 2024

@author: akshayjacobthomas
"""

import jax.numpy as jnp
import jax
from jax import random
from math import pi
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt
from flax.linen.initializers import lecun_normal, glorot_normal
from flax.linen.initializers import zeros
from typing import Any, Callable, Sequence  # Importing these for data classes data fields
import optax
import optax
from time import perf_counter
from jax import random
import pickle
from functools import partial


def get_solution(x_min=0., x_max=1., total_time=1., n_points=20):
    
    x_dim = 1  # dimension in spatial domain nad not time
    t_dim = 1  # dimension in time domain
    x_key,= random.split(random.PRNGKey(1234), num=1)
    #t_sample = jnp.array([[time]]) # single time sample 
    
    # make points to plit
    X_init = jnp.linspace(x_min, x_max, n_points)
    T_init= jnp.linspace(0., total_time, n_points)
    xx, yy = jnp.meshgrid(X_init, T_init)
    x_volume = jnp.concatenate((xx.reshape(-1)[:,None], yy.reshape(-1)[:,None]), axis=1)
    
    xv_samples = x_volume
    # make vecetorizable functions for temperature
    def make_analytical_solution(n_, L, alpha):
        
        def analytical_solution(x, t):
            def evaluate_term(n):
                const = -2.*L*L/((n**3)*(pi**3))
                Bi = 1.*const* (-2. + 2.* jnp.cos(n*pi) + n*pi* jnp.sin(n*pi))

                return Bi* jnp.sin(n*pi*x/L)*jnp.exp(-1.*((n*pi/L)**2)*alpha*t)

            return jnp.sum(jax.vmap(evaluate_term)(n_))

        return analytical_solution

    temperature = make_analytical_solution(np.linspace(1, 500, num=500), x_max, 5e-2)
    values = jax.vmap(jax.vmap(temperature, (0, None)), (None, 0))(xv_samples[:,0], xv_samples[:,1])
    
    
    return values, xv_samples[:,1], xv_samples[:,0]

    
def get_solution_plot(x_min=0., x_max=1., total_time=1., n_points=250):
    
    x_dim = 1  # dimension in spatial domain nad not time
    t_dim = 1  # dimension in time domain
    x_key,= random.split(random.PRNGKey(1234), num=1)
    #t_sample = jnp.array([[time]]) # single time sample 
    
    # make points to plit
    X_init = jnp.linspace(x_min, x_max, n_points)
    T_init= jnp.linspace(0., total_time, n_points)
    xx, yy = jnp.meshgrid(X_init, T_init)
    x_volume = jnp.concatenate((xx.reshape(-1)[:,None], yy.reshape(-1)[:,None]), axis=1)
    
    xv_samples = x_volume
    # make vecetorizable functions for temperature
    def make_analytical_solution(n_, L, alpha):
        
        def analytical_solution(x, t):
            def evaluate_term(n):
                const = -2.*L*L/((n**3)*(pi**3))
                Bi = 1.*const* (-2. + 2.* jnp.cos(n*pi) + n*pi* jnp.sin(n*pi))

                return Bi* jnp.sin(n*pi*x/L)*jnp.exp(-1.*((n*pi/L)**2)*alpha*t)

            return jnp.sum(jax.vmap(evaluate_term)(n_))

        return analytical_solution

    temperature = make_analytical_solution(np.linspace(1, 500, num=500), x_max, 5e-2)
    values = jax.vmap(temperature)(xv_samples[:,0], xv_samples[:,1])
    
    
    return values, xv_samples[:,1], xv_samples[:,0]