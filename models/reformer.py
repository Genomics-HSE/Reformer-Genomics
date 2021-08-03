import trax
import trax.layers as tl
from trax.fastmath import numpy as jnp

from .custom_encoder_block import EncoderBlock
from .custom_positional_encoding import PositionalEncoding


def ReformerModel(d_model, d_ff, n_heads, attention_type, dropout, ff_activation,
                  ff_dropout, n_layers, max_len, filters, kernel_size, stride, mode='train'):
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
        tl.Conv1d(filters=filters, kernel_size=kernel_size, stride=stride, padding="SAME"),
        PositionalEncoding(max_len=max_len, mode=mode),
        tl.Dense(d_model),
        tl.Dup(),
        tl.ReversibleSerial(encoder_blocks),
        tl.Concatenate(),
        tl.Dense(32),
        tl.LogSoftmax(),
    )
    
    return encoder


def Printer():
    from trax.fastmath import numpy as jnp
    layer_name = "Printer"
    
    def func(x, y):
        return x, y
    
    return tl.Fn(layer_name, func, n_out=2)


def ExpandDim():
    layer_name = "ExpandDim"
    
    def func(x):
        x = x.astype(jnp.float32)
        return jnp.expand_dims(x, axis=-1)  # , jnp.expand_dims(y, axis=2)
    
    return tl.Fn(layer_name, func, n_out=1)
