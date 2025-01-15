
"""
Created on Mon Feb 20 15:18:52 2024

@author: akshayjacobthomas

"""
# contains samplers for additive problems using PINN
from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import random, pmap, local_device_count

from torch.utils.data import Dataset


class BaseSampler(Dataset):
    def __init__(self, batch_size, rng_key=random.PRNGKey(1234)):
        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = local_device_count()

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        batch = self.data_generation(keys)
        return batch

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")

# base sampler for future sequantial sampling
# might have to change the bondary sampler to be a generator instead of iterator
class SeqBaseSampler():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    #@partial(pmap, static_broadcasted_argnums=(0,))
    def __call__(self, step, time):
        "Generate one batch of data"
        keys = random.split(random.PRNGKey(step), local_device_count())
        batch = self.data_generation(keys, time)
        return batch

    def data_generation(self, key, time):
        raise NotImplementedError("Subclasses should implement this!")


class StepIndexSampler(BaseSampler):
    def __init__(self, n_test:int, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.n_test=n_test
        self.dim = 1
        
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        batch = random.categorical(key,
                                   logits=jnp.ones(int(self.n_test)),
                                   axis=0,
                                   shape=(self.batch_size, 1))
        return batch


class  UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )

        return batch

class SeqInitialSampler(BaseSampler):

    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        time_samples = jnp.zeros((self.batch_size, 1))

        batch = jnp.concatenate((time_samples, batch), axis=1)[0]
        return batch


class SeqBoundarySampler(BaseSampler):

    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(124)):
        super().__init__(batch_size, rng_key)

        self.dom = dom
        self.dim = dom.shape[0]
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )

        return batch



class SeqCollocationSampler(SeqBaseSampler):
    """Samples for sequential collocation sampling"""

    def __init__(self, batch_size, init_length: jnp.array,
                 velocity: jnp.array, bw, rng_key = random.PRNGKey(1234)):
        """The velocity is a vector since velocity is a vector, but we will
        use only one direction - which is the first component"""

        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bw
        self.dim = velocity.shape[0]


    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0]*time + self.init_length[0]
        width_updated = self.velocity[1]*time + self.bead_width

        x_batch = random.uniform(key,
                               shape = (self.batch_size, 1),
                               minval = jnp.array([0.]),
                               maxval = jnp.array([length_updated]))

        y_batch = random.uniform(key+10,
                        shape = (self.batch_size, 1),
                        minval = jnp.array([0.]),
                        maxval = jnp.array([width_updated]))

        volume_batch = jnp.concatenate((x_batch, y_batch), axis=1)
        batch = jnp.concatenate((time*jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

class SeqNeumanCollocationSampler_B1(SeqBaseSampler):
    
    """Samples in sequiential collocation sampling for Natural BCs"""
    
    def __init__(self, batch_size, init_length: jnp.array,
                 velocity: jnp.array, bw, rng_key = random.PRNGKey(1234)):
        """The velocity is a vector since velocity is a vector, but we will
        use only one direction - which is the first component"""

        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bw
        self.dim = velocity.shape[0]
        
    @partial(pmap, static_broadcasted_argnums=(0,))    
    def data_generation(self, key, time):
        
        length_updated = self.velocity[0]*time + self.init_length[0]
        
        
        x_batch = random.uniform(key,
                                shape = (self.batch_size, 1),
                                minval = jnp.array([0.]),
                                maxval = jnp.array([length_updated]))
        y_batch = random.uniform(key,
                                shape = (self.batch_size, 1),
                                minval = jnp.array([1.]),
                                maxval = jnp.array([1.]))

        
        volume_batch = jnp.concatenate((x_batch, y_batch), axis=1)
        batch = jnp.concatenate((time*jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        
        return batch
    
class SeqNeumanCollocationSampler_B2(SeqBaseSampler):
    
    """Samples in sequiential collocation sampling for Natural BCs"""
    
    def __init__(self, batch_size, init_length: jnp.array,
                 velocity: jnp.array, bw, rng_key = random.PRNGKey(1234)):
        """The velocity is a vector since velocity is a vector, but we will
        use only one direction - which is the first component"""

        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bw
        self.dim = velocity.shape[0]
        
    @partial(pmap, static_broadcasted_argnums=(0,))    
    def data_generation(self, key, time):
        
        length_updated = self.velocity[0]*time + self.init_length[0]
        
        
        x_batch = random.uniform(key,
                                shape = (self.batch_size, 1),
                                minval = jnp.array([0.]),
                                maxval = jnp.array([length_updated]))
        y_batch = random.uniform(key,
                                shape = (self.batch_size, 1),
                                minval = jnp.array([0.]),
                                maxval = jnp.array([0.]))

        
        volume_batch = jnp.concatenate((x_batch, y_batch), axis=1)
        batch = jnp.concatenate((time*jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        
        return batch
    
class NeumannInitialSampler(SeqBaseSampler):
    """ Generates samples that maintain the unactivated boundaries at depositio
    temperature. Will sampler in higher dimensions later"""
    
    def __init__(self, batch_size, init_length: jnp.array,
                 velocity: jnp.array, bw, rng_key = random.PRNGKey(1234)):
        
        """The velocity is a vector since velocity is a vector, but we will
        use only one direction - which is the first component"""

        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bw
        self.dim = velocity.shape[0]
        
    @partial(pmap, static_broadcasted_argnums=(0,))    
    def data_generation(self, key, time):
        
        length_updated = self.velocity[0]*time + self.init_length[0]
        
        
        x_batch = random.uniform(key,
                                shape = (self.batch_size, 1),
                                minval = jnp.array([length_updated]),
                                maxval = jnp.array([1.0]))
        y_batch1 = random.uniform(key,
                                shape = (int(self.batch_size/2), 1),
                                minval = jnp.array([1.]),
                                maxval = jnp.array([1.]))
        
        
        y_batch2 = random.uniform(key,
                                shape = (int(self.batch_size/2), 1),
                                minval = jnp.array([0.]),
                                maxval = jnp.array([0.]))
        
        y_batch = jnp.concatenate((y_batch1, y_batch2), axis=0)
        volume_batch = jnp.concatenate((x_batch, y_batch), axis=1)
        batch = jnp.concatenate((time*jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        
        return batch
            


class SeqInitialBoundarySampler(SeqBaseSampler):
    """ Generates samples that maintain the  active boundaries at depositio
    temperature. Will sampler in higher dimensions later"""
    
    
    def __init__(self, batch_size, init_length: jnp.array,
                 velocity: jnp.array, bw, rng_key = random.PRNGKey(1234)):
        """The velocity is a vector since velocity is a vector, but we will
        use only one direction - which is the first component"""

        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bw
        self.dim = velocity.shape[0]
        
    @partial(pmap, static_broadcasted_argnums=(0,))    
    def data_generation(self, key, time):
        
        length_updated = self.velocity[0]*time + self.init_length[0]
        
        
        x_batch = random.uniform(key,
                                shape = (self.batch_size, 1),
                                minval = jnp.array([length_updated]),
                                maxval = jnp.array([length_updated]))
        y_batch = random.uniform(key,
                                shape = (self.batch_size, 1),
                                minval = jnp.array([0.0]),
                                maxval = jnp.array([1.]))

        
        volume_batch = jnp.concatenate((x_batch, y_batch), axis=1)
        batch = jnp.concatenate((time*jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        
        return batch
    
    
class LD_HaltonSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        
        super().__init__(batch_size, rng_key)
        self.dom = dom
        if dom[0,1]>1.:
            raise NotImplementedError("Halton sequence not suported in the specified domain")
        self.dim = dom.shape[0]

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        u0 = random.uniform(
            key,
            shape=(1, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        
        batch= jnp.mod(u0+ jnp.arange(
            1, self.batch_size+1, 1)/self.batch_size, 1)

        return batch.reshape(1,self.batch_size,1)[0]
        
