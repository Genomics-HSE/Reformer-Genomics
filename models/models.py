import trax
import trax.layers as tl
from trax.fastmath import numpy as jnp
import copy

from .custom_encoder_block import EncoderBlock
from .custom_positional_encoding import PositionalEncoding


def ReformerModel(embedding_obj, d_model, d_ff, n_heads, attention_type, dropout, ff_activation,
                  ff_dropout, n_layers, max_len, output_dim, mode='train'):
    encoder_blocks = [EncoderBlock(d_model=d_model,
                                   d_ff=d_ff,
                                   n_heads=n_heads,
                                   attention_type=attention_type,
                                   dropout=dropout,
                                   ff_activation=ff_activation,
                                   ff_dropout=ff_dropout,
                                   mode=mode
                                   ) for _ in range(n_layers)]
    
    encoder = tl.Serial(
        embedding_obj,
        #PositionalEncoding(max_len=max_len, mode=mode),
        tl.PositionalEncoding(mode=mode),
        tl.Dense(d_model),
        tl.Dup(),
        tl.ReversibleSerial(encoder_blocks),
        tl.Concatenate(),
        tl.Dense(output_dim),
        tl.Relu(),
        tl.Dense(output_dim),
        tl.LogSoftmax(),
    )
    
    return encoder


def CNNEmbedding(filters, kernel_size, stride):
    emb = tl.Conv1d(filters=filters, kernel_size=kernel_size, stride=stride, padding="SAME"),
    return emb


def Embedding(vocab_size, d_model):
    emb = tl.Embedding(vocab_size, d_model)
    return emb


def GruModel(d_model, vocab_size, n_units, mode):
    model = tl.Serial(
        tl.Embedding(vocab_size, d_model),
        Bidirectional(tl.GRU(n_units, mode)),
        tl.Dense(d_model),
        tl.Relu(),
        tl.Dense(d_model),
        tl.LogSoftmax()
    )
    return model


def Bidirectional(forward_layer, axis=1, merge_layer=tl.Concatenate()):
    """Bidirectional combinator for RNNs.
    Args:
      forward_layer: A layer, such as `trax.layers.LSTM` or `trax.layers.GRU`.
      axis: a time axis of the inputs. Default value is `1`.
      merge_layer: A combinator used to combine outputs of the forward
        and backward RNNs. Default value is 'trax.layers.Concatenate'.
    Example:
        Bidirectional(RNN(n_units=8))
    """
    backward_layer = copy.deepcopy(forward_layer)
    flip = tl.base.Fn('_FlipAlongTimeAxis', lambda x: jnp.flip(x, axis=axis))
    backward = tl.Serial(
        flip,
        backward_layer,
        flip,
    )
    
    return tl.Serial(
        tl.Branch(forward_layer, backward),
        merge_layer,
    )


def Printer():
    from trax.fastmath import numpy as jnp
    layer_name = "Printer"
    
    def func(x, y):
        print(x.shape, y.shape)
        return x, y
    
    return tl.Fn(layer_name, func, n_out=2)


def ExpandDim():
    layer_name = "ExpandDim"
    
    def func(x):
        return jnp.expand_dims(x, axis=-1)  # , jnp.expand_dims(y, axis=2)
    
    return tl.Fn(layer_name, func, n_out=1)
