import numpy as np

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers.assert_shape import assert_shape

from trax.layers import initializers as init


@assert_shape('...d->...d')
class PositionalEncoding(base.Layer):
    """Implements bare positional encoding.
  
    Positional encoding includes a kind of dropout, if the layer is created in
    ``'train'`` mode with a nonzero ``dropout`` value. For such a layer, on each
    forward pass a subset of sequence positions selected at random will *not*
    receive positional marking.
    """
    
    def __init__(self, max_len=2048, dropout=0.0, dropout_broadcast_dims=(-2,),
                 use_bfloat16=False, start_from_zero_prob=1.0,
                 max_offset_to_add=0, d_feature=None, mode='train'):
        """Creates a :py:class:`PositionalEncoding` instance in a given mode.
    
        Args:
          max_len: Maximum input sequence length.
          dropout: Probability of *not* adding positional encoding to a sequence
              position. Applies only if layer is created in ``'train'`` mode.
          dropout_broadcast_dims: Axes along which dropout mask values are
              broadcast rather than individually set at random.
          use_bfloat16: If ``True``, use bfloat16 weights instead of the default
            float32; this can save memory but may (rarely) lead to numerical issues.
          start_from_zero_prob: how often to start from 0 during training,
              (if 1.0, we always start from position 0, if less, we randomize).
          max_offset_to_add: maximum offset to add to the positions during training
            when randomizing; this offset plus input length must still be less than
            max_len for all training examples.
          d_feature: int or None; have this dimension for embeddings + shared FF if
            not None.
          mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
        """
        super().__init__()
        self._max_len = max_len
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')
        if mode == 'train':
            self._dropout = dropout
        else:
            self._dropout = 0.0
        self._dropout_broadcast_dims = dropout_broadcast_dims
        self._use_bfloat16 = use_bfloat16
        self._start_from_zero_prob = start_from_zero_prob
        self._max_offset_to_add = max_offset_to_add
        self._mode = mode
        self._d_feature = d_feature
    
    def forward(self, inputs):
        """Returns the input activations, with added positional information."""
        weights = self.weights
        if self._d_feature is not None and self._mode != 'predict':
            weights, ff = weights
            weights = jnp.dot(weights[:inputs.shape[1], :], ff)
        if len(weights.shape) < 3:  # old checkpoints have 1 in first dim already
            weights = weights[None, :, :]  # [1, self._max_len, d_feature]
        if self._mode != 'predict':
            x = inputs
            symbol_size = jnp.shape(x)[1]
            if self._mode != 'train' or self._start_from_zero_prob >= 1.0:
                px = weights[:, :symbol_size, :]
            else:
                rng1, rng2 = fastmath.random.split(self.rng, 2)
                start = fastmath.random.randint(rng1, (), 0, self._max_offset_to_add)
                start_from_zero = fastmath.random.uniform(rng2, (), jnp.float32, 0, 1)
                start = jnp.where(start_from_zero < self._start_from_zero_prob,
                                  jnp.zeros((), dtype=jnp.int32), start)
                px = fastmath.dynamic_slice_in_dim(weights, start, symbol_size,
                                                   axis=1)
            if self._dropout == 0:
                return x + px
            else:
                noise_shape = list(px.shape)
                for dim in self._dropout_broadcast_dims:
                    noise_shape[dim] = 1
                keep_prob = 1.0 - self._dropout
                keep = fastmath.random.bernoulli(self.rng, keep_prob,
                                                 tuple(noise_shape))
                multiplier = keep.astype(x.dtype) / keep_prob
                return x + px * multiplier
        else:
            if self._dropout != 0:
                raise ValueError(f'In predict mode, but dropout rate '
                                 f'({self._dropout}) is not zero.')
            
            # State in this class is only used for fast inference. In that case,
            # the model is called with consecutive elements position-by-position.
            # This positional encoding layer stores the index of the current
            # position and increments it on each call.
            emb = fastmath.dynamic_slice_in_dim(
                weights, self.state, inputs.shape[1], axis=1)
            self.state += inputs.shape[1]
            return inputs + emb
    
    def init_weights_and_state(self, input_signature):
        """Randomly initializes the positional encoding vectors.
    
        Args:
          input_signature: :py:class:`ShapeDtype` instance characterizing the input
              this layer should compute on.
        """
        d_feature = input_signature.shape[-1]
        if self._d_feature is not None:
            d_feature = self._d_feature
        pe = np.zeros((self._max_len, d_feature), dtype=np.float32)
        position = np.arange(0, self._max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_feature, 2) * -(np.log(self._max_len) / d_feature))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)  # [self._max_len, d_feature]
        if self._use_bfloat16:
            pe = pe.astype(jnp.bfloat16)
        w = jnp.array(pe)  # Trainable parameters, initialized above.
        if self._d_feature is not None:
            ff = init.GlorotUniformInitializer()(
                (d_feature, input_signature.shape[-1]), self.rng)
            self.weights = w, ff
        else:
            self.weights = w
        if self._mode == 'predict':
            self.state = jnp.zeros((), dtype=jnp.int32)


# @assert_shape('...d->...d')
# class PositionalEncoding(base.Layer):
#     """Implements bare positional encoding.
#
#     Positional encoding includes a kind of dropout, if the layer is created in
#     ``'train'`` mode with a nonzero ``dropout`` value. For such a layer, on each
#     forward pass a subset of sequence positions selected at random will *not*
#     receive positional marking.
#     """
#
#     def __init__(self, max_len=2048, dropout=0.0, dropout_broadcast_dims=(-2,),
#                  use_bfloat16=False, start_from_zero_prob=1.0,
#                  max_offset_to_add=0, d_feature=None, mode='train'):
#         """Creates a :py:class:`PositionalEncoding` instance in a given mode.
#
#         Args:
#           max_len: Maximum input sequence length.
#           dropout: Probability of *not* adding positional encoding to a sequence
#               position. Applies only if layer is created in ``'train'`` mode.
#           dropout_broadcast_dims: Axes along which dropout mask values are
#               broadcast rather than individually set at random.
#           use_bfloat16: If ``True``, use bfloat16 weights instead of the default
#             float32; this can save memory but may (rarely) lead to numerical issues.
#           start_from_zero_prob: how often to start from 0 during training,
#               (if 1.0, we always start from position 0, if less, we randomize).
#           max_offset_to_add: maximum offset to add to the positions during training
#             when randomizing; this offset plus input length must still be less than
#             max_len for all training examples.
#           d_feature: int or None; have this dimension for embeddings + shared FF if
#             not None.
#           mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
#         """
#         super().__init__()
#         self._max_len = max_len
#         if dropout >= 1.0:
#             raise ValueError('Dropout rates must be lower than 1.')
#         if mode == 'train':
#             self._dropout = dropout
#         else:
#             self._dropout = 0.0
#         self._dropout_broadcast_dims = dropout_broadcast_dims
#         self._use_bfloat16 = use_bfloat16
#         self._start_from_zero_prob = start_from_zero_prob
#         self._max_offset_to_add = max_offset_to_add
#         self._mode = mode
#         self._d_feature = d_feature
#
#     def forward(self, inputs):
#         """Returns the input activations, with added positional information."""
#         weights = self.weights
#         if self._d_feature is not None and self._mode != 'predict':
#             weights, ff = weights
#             weights = jnp.dot(weights[:inputs.shape[1], :], ff)
#         if len(weights.shape) < 3:  # old checkpoints have 1 in first dim already
#             weights = weights[None, :, :]  # [1, self._max_len, d_feature]
#         if self._mode != 'predict':
#             x = inputs
#             symbol_size = jnp.shape(x)[1]
#             if self._mode != 'train' or self._start_from_zero_prob >= 1.0:
#                 px = weights[:, :symbol_size, :]
#             else:
#                 rng1, rng2 = fastmath.random.split(self.rng, 2)
#                 start = fastmath.random.randint(rng1, (), 0, self._max_offset_to_add)
#                 start_from_zero = fastmath.random.uniform(rng2, (), jnp.float32, 0, 1)
#                 start = jnp.where(start_from_zero < self._start_from_zero_prob,
#                                   jnp.zeros((), dtype=jnp.int32), start)
#                 px = fastmath.dynamic_slice_in_dim(weights, start, symbol_size,
#                                                    axis=1)
#             if self._dropout == 0:
#                 return jnp.array(jnp.concatenate((x, px), axis=-1))
#             else:
#                 noise_shape = list(px.shape)
#                 for dim in self._dropout_broadcast_dims:
#                     noise_shape[dim] = 1
#                 keep_prob = 1.0 - self._dropout
#                 keep = fastmath.random.bernoulli(self.rng, keep_prob,
#                                                  tuple(noise_shape))
#                 multiplier = keep.astype(x.dtype) / keep_prob
#                 return jnp.array(jnp.concatenate((x, px * multiplier), axis=-1))
#         else:
#             if self._dropout != 0:
#                 raise ValueError(f'In predict mode, but dropout rate '
#                                  f'({self._dropout}) is not zero.')
#
#             # State in this class is only used for fast inference. In that case,
#             # the model is called with consecutive elements position-by-position.
#             # This positional encoding layer stores the index of the current
#             # position and increments it on each call.
#             emb = fastmath.dynamic_slice_in_dim(
#                 weights, self.state, inputs.shape[1], axis=1)
#             self.state += inputs.shape[1]
#             return jnp.array(jnp.concatenate((inputs, emb), axis=-1))
#
#     def init_weights_and_state(self, input_signature):
#         """Randomly initializes the positional encoding vectors.
#
#         Args:
#           input_signature: :py:class:`ShapeDtype` instance characterizing the input
#               this layer should compute on.
#         """
#         d_feature = input_signature.shape[-1] + 1  # NB!
#         if self._d_feature is not None:
#             d_feature = self._d_feature
#         pe = np.zeros((self._max_len, d_feature), dtype=np.float32)
#         position = np.arange(0, self._max_len)[:, np.newaxis]
#         div_term = np.exp(
#             np.arange(0, d_feature, 2) * -(np.log(self._max_len) / d_feature))  #NB
#         pe[:, 0::2] = np.sin(position * div_term)
#         pe[:, 1::2] = np.cos(position * div_term)  # [self._max_len, d_feature]
#         if self._use_bfloat16:
#             pe = pe.astype(jnp.bfloat16)
#         w = jnp.array(pe)  # Trainable parameters, initialized above.
#         if self._d_feature is not None:
#             ff = init.GlorotUniformInitializer()(
#                 (d_feature, input_signature.shape[-1]), self.rng)
#             self.weights = w, ff
#         else:
#             self.weights = w
#         if self._mode == 'predict':
#             self.state = jnp.zeros((), dtype=jnp.int32)
