import random
from os.path import join
from typing import Any, List
import gin
import trax
import trax.layers as tl
from trax.supervised import training
from trax.fastmath import numpy as jnp
import scipy
import matplotlib.pyplot as plt

from .viz import make_coalescent_heatmap
from .data_gen_np import get_generator, get_list_of_generators


@gin.configurable
def train_model(model, train_gen, comet_exp, lr, n_warmup_steps, n_steps_per_checkpoint, output_dir, n_steps):
    lr_schedule = trax.lr.warmup_and_rsqrt_decay(
        n_warmup_steps=n_warmup_steps, max_value=lr)
    train_task = training.TrainTask(
        labeled_data=train_gen,
        loss_layer=tl.CategoryCrossEntropy(),  # KL_DIV()
        optimizer=trax.optimizers.Adam(lr),
        lr_schedule=lr_schedule,
        n_steps_per_checkpoint=n_steps_per_checkpoint
    )
    
    eval_task = training.EvalTask(
        labeled_data=train_gen,
        metrics=[KL_DIV()]
    )
    
    loop = training.Loop(model=model,
                         tasks=train_task,
                         eval_tasks=eval_task,
                         output_dir=output_dir,
                         eval_at=lambda x: x % 10 == 0)
    
    with comet_exp.train():
        loop.run(n_steps=n_steps)


@gin.configurable
def predict(model, model_path, data_gen, num_genomes, plot_dir, plot_length=-1, genome_length=1, min_length_to_plot=300000):
    model.init_from_file(model_path, weights_only=True)
    
    for i in range(num_genomes):
        data = next(data_gen)
        X, y = data
        
        predictions = model(X)
        predictions = jnp.exp(jnp.squeeze(predictions, 0).T)
        y = jnp.squeeze(y, 0)
        
        if plot_length == -1:
            plot_length = len(y)
        num_plots = int(len(y) / plot_length)
        
        ptr = 0
        for j in range(num_plots):
            figure = make_coalescent_heatmap("", (predictions[:, ptr:ptr + plot_length], y[ptr:ptr + plot_length]))
            plt.savefig(join(plot_dir, "plots", str(i) + "_" + str(j)))
            plt.close(figure)
            ptr += plot_length


@gin.configurable
def get_trax_generator(num_genomes, genome_length, num_generators,
                       random_seed, batch_size):
    generators = get_list_of_generators(num_genomes=num_genomes, genome_length=genome_length,
                                        num_generators=num_generators,
                                        random_seed=random_seed)
    
    generator = MixGenerator(generators, num_genomes)
    
    serial_generator = trax.data.Serial(
        trax.data.Batch(batch_size)
    )(generator)
    
    return serial_generator


class MixGenerator:
    def __init__(self, list_of_generators: List[Any], examples_in_one_gen: int):
        self.list_of_generators = list_of_generators
        self.examples_in_one_gen = examples_in_one_gen
        self.total_examples = len(list_of_generators) * self.examples_in_one_gen
        self.counter = {i: 0 for i in range(len(list_of_generators))}
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < self.total_examples:
            random_gen_index = random.randint(0, len(self.list_of_generators)-1)
            generator = self.list_of_generators[random_gen_index]
            sample = next(generator)
            
            self.counter[random_gen_index] += 1
            self.i += 1
            
            if self.counter[random_gen_index] == self.examples_in_one_gen:
                self.list_of_generators.remove(generator)
            
            return sample
        else:
            raise StopIteration


@gin.configurable(denylist=['logpred', 'target'])
def kl_div(logpred, target, eps=jnp.finfo(jnp.float32).eps):
    """Calculate KL-divergence."""
    return jnp.sum(target * (jnp.log(target + eps) - logpred))


kl_div = tl.layer_configure(kl_div)


def KL_DIV():
    def f(model_output, targets):
        if len(targets.shape) < 3:
            # one hot encoding
            targets = one_hot_encoding_numpy(targets, model_output.shape[-1])
        divergence = kl_div(model_output, targets)
        return jnp.average(divergence)
    
    return tl.base.Fn("KL_DIV", f)


def one_hot_encoding_numpy(y_data, num_class):
    """
    
    :param batch_data: (batch_size, seq_len)
    :return:
    """
    return jnp.arange(num_class) == y_data[..., None].astype(jnp.float32)
