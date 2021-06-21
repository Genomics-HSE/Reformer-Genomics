import trax
import trax.layers as tl
from trax.fastmath import numpy as jnp
from trax.models.research import configurable_transformer as ct


def EncoderBlock(d_model, d_ff, n_heads, attention_type, dropout, ff_activation,
                 ff_dropout, ff_use_sru=0, ff_chunk_size=0, ff_sparsity=0,
                 attention_chunk_size=0, center_layernorm=True,
                 use_bfloat16=False, use_two_swaps_per_block=True,
                 mode='train'):
    """Returns a list of layers that implements a Reformer encoder block.
    The input to the layer is a pair, (activations, mask), where the mask was
    created from the original source tokens to prevent attending to the padding
    part of the input.
    Args:
      d_model: int:  depth of embedding
      d_ff: int: depth of feed-forward layer
      n_heads: int: number of attention heads
      attention_type: subclass of tl.BaseCausalAttention: attention class to use
      dropout: float: dropout rate (how much to drop out)
      ff_activation: the non-linearity in feed-forward layer
      ff_dropout: the dropout rate in feed-forward layer
      ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
      ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
      ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
      attention_chunk_size: int, if > 0 run attention chunked at this size
      center_layernorm: whether to use centering in LayerNorm (default) or if
        to skip it, which is known as RMS normalization.
      use_bfloat16: whether to use bfloat16 for weights (default: False)
      use_two_swaps_per_block: bool, if True use two reversible swaps in Encoder
        block, otherwise use only one swap.
      mode: str: 'train' or 'eval'
    Returns:
      A list of layers that maps (activations, mask) to (activations, mask).
    """
    if mode == 'predict':
        # Mode 'predict' means that the decoder should be run one token at a time.
        # The encoder only ever runs over full sequences, which is why it's switched
        # to 'eval' mode instead.
        mode = 'eval'
    
    def _Attn():
        return ct.ApplyAttentionLayer(
            attention_type=attention_type, d_model=d_model, n_heads=n_heads,
            d_qk=d_model // n_heads, d_v=d_model // n_heads, masked=False, causal=False,
            attention_dropout=dropout, output_dropout=dropout,
            attention_chunk_size=attention_chunk_size, mode=mode)  # NB!
    
    def _FF():
        return ct.FeedForwardWithOptions(
            d_model, d_ff, dropout, [-2], ff_activation, ff_dropout,
            ff_chunk_size, ff_use_sru, ff_sparsity, center_layernorm,
            mode, use_bfloat16)
    
    # TODO(lukaszkaiser): refactor efficient attention layers to unify the API
    # If we're using standard attention, we need to pass reshaped mask and not
    # return the mask to be compatible with the EfficientAttention API.
    attention = _Attn()
    if attention.n_out == 2:
        attention = tl.Serial(
            tl.Parallel([], _InsertAxes12()),
            attention,
            tl.Select([0], n_in=2)
        )
    
    def _attention_half_residual():
        return [
            tl.ReversibleHalfResidual(tl.LayerNorm(center=center_layernorm),
                                      attention_layer=attention,
                                      name='ReversibleHalfResidualEncoderAttn'),
            tl.ReversibleSwap()
        ]
    
    def _feed_forward():
        layers = [
            tl.ReversibleHalfResidual(_FF(),
                                      name='ReversibleHalfResidualEncoderFF')
        ]
        if use_two_swaps_per_block:
            layers.append(tl.ReversibleSwap())
        return layers
    
    return _attention_half_residual() + _feed_forward()


def _InsertAxes12():
    """Returns a layer that inserts two internal size-1 axes into an array."""
    return tl.Fn('InsertAxes12',
                 lambda x: jnp.reshape(x, (x.shape[0], 1, 1, x.shape[1])))



