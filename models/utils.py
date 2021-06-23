import gin
import trax
from .data_gen_np import get_generator
from trax.supervised import training


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
        loss_layer=trax.layers.CategoryCrossEntropy(),
        optimizer=trax.optimizers.Adam(lr),
        lr_schedule=lr_schedule,
        n_steps_per_checkpoint=n_steps_per_checkpoint
    )
    
    loop = training.Loop(model,
                         train_task,
                         output_dir=output_dir)
    
    with comet_exp.train():
        loop.run(n_steps=n_steps)
