import comet_ml

import gin
import trax
import trax.layers as tl
from trax.supervised import training

from data_gen_np import get_generator
from custom_positional_encoding import PositionalEncoding
from custom_encoder_block import EncoderBlock, ExpandDim


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


def ReformerModel(d_model, d_ff, n_heads, attention_type, dropout, ff_activation,
                  ff_dropout, n_layers, max_len, mode='train'):
    encoder_blocks = [EncoderBlock(d_model=d_model,
                                   d_ff=d_ff,
                                   n_heads=n_heads,
                                   attention_type=attention_type,
                                   dropout=dropout,
                                   ff_activation=ff_activation,
                                   ff_dropout=ff_dropout,
                                   mode=mode
                                   ) for _ in range(n_layers)]
    
    encoder = trax.layers.Serial(
        ExpandDim(),
        Printer(),
        PositionalEncoding(max_len=max_len, mode=mode),
        tl.Dense(d_model),
        tl.Dup(),
        tl.ReversibleSerial(encoder_blocks),
        tl.Concatenate(),
        tl.Dense(d_model),
        tl.LogSoftmax()
    )
    
    return encoder


def Printer():
    from trax.fastmath import numpy as jnp
    layer_name = "Printer"
    
    def func(x, y):
        return x, y
    
    return tl.Fn(layer_name, func, n_out=2)


ReformerModel = trax.models.model_configure(ReformerModel)


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


if __name__ == '__main__':
    gin.parse_config_file("config.gin")
    
    train_gen = get_trax_generator()
    comet_exp = comet_ml.Experiment()
    
    train(model=ReformerModel(), train_gen=train_gen, comet_exp=comet_exp)
