from os.path import join
import gin
import trax
import trax.layers as tl
from trax.supervised import training
from trax.fastmath import numpy as jnp
import scipy
import matplotlib.pyplot as plt

from .viz import make_coalescent_heatmap
from .data_gen_np import get_generator


@gin.configurable
def get_trax_generator(num_genomes, genome_length, num_generations,
                       random_seed, num_demography, batch_size):
    generator = get_generator(num_genomes=num_genomes, genome_length=genome_length, num_generators=num_generations,
                              random_seed=random_seed)
    generator = next(generator)
    serial_generator = trax.data.Serial(
        trax.data.Batch(batch_size)
    )(generator)
    
    return serial_generator


@gin.configurable
def train(model, train_gen, comet_exp, lr, n_warmup_steps, n_steps_per_checkpoint, output_dir, n_steps):
    lr_schedule = trax.lr.warmup_and_rsqrt_decay(
        n_warmup_steps=n_warmup_steps, max_value=lr)
    train_task = training.TrainTask(
        labeled_data=train_gen,
        loss_layer=KL_DIV(),
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
def predict(model, model_path, data_generator, num_genomes, genome_length=1, min_length_to_plot=300000):
    model.init_from_file(model_path, weights_only=True)
    
    for i in range(num_genomes):
        data = next(data_generator)
        X, y = data
        
        predictions = model(X)
        predictions = jnp.exp(jnp.squeeze(predictions, 0).T)
        y = jnp.squeeze(y, 0)
        
        figure = make_coalescent_heatmap("", (predictions, y))
        plt.savefig(join("output/plots", str(i)))
        plt.close(figure)


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
