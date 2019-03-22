import numpy as np
from renom.config import precision

class Initializer(object):
    """Base class of initializer.

    When the initialization of parameterized layer class,
    dense, conv2d, lstm ... , you can select the initialization method
    changing the initializer class as following example.

    Example:
        >>> import renom as rm
        >>> from renom.utility.initializer import GlorotUniform
        >>>
        >>> layer = rm.Dense(output_size=2, input_size=2, initializer=GlorotUniform())
        >>> print(layer.params.w)
        [[-0.55490332 -0.14323548]
         [ 0.00059367 -0.28777076]]

    """

    def __call__(self, shape):
        raise NotImplementedError


class Orthogonal(Initializer):

    '''Orthogonal initializer.
    Initialize parameters using orthogonal initialization.

    .. [1] Andrew M. Saxe, James L. McClelland, Surya Ganguli https://arxiv.org/abs/1312.6120
       Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    '''

    def __init__(self, gain=1.0):
        super(Orthogonal, self).__init__()
        self.gain = gain

    def __call__(self, shape):
        c_shape = (shape[0], np.prod(shape[1:]))
        X = np.random.random(c_shape)-0.5
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        res = U if U.shape==c_shape else Vt

        return (res.reshape(shape) * self.gain).astype(precision)
