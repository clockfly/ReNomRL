import numpy as np
from renom import Node


class DiscreteNodeChooser(object):
    """
    Base class of choosing action nodes in discrete action space.
    """

    def __call__(self):
        raise "please override this function"

    def _trasform_node_2_numpy(self, node_var):
        """
        this function changes node variables to numpy
        """

        if isinstance(node_var, Node):
            node_var = node_var.as_ndarray()

        assert len(node_var.shape) > 1, "The node_var must be more than 2D"

        return node_var


class MaxNodeChooser(DiscreteNodeChooser):
    """
    **Max Node Chooser**

    Chooses max node index. Returns 1D numpy list.
    If the length is 1, it returns int.
    """

    def __call__(self, node_var):

        node_var = self._trasform_node_2_numpy(node_var)

        max_list = np.argmax(node_var, axis=1)
        if len(max_list) == 1:
            return int(max_list)
        else:
            return max_list


class ProbNodeChooser(DiscreteNodeChooser):
    """
    **Probability Node Chooser**

    Chooses node index based on its output probability. Returns 1D numpy list.
    If the length is 1, it returns int.
    """

    def __call__(self, node_var):

        node_var = self._trasform_node_2_numpy(node_var)

        norm = np.sum(node_var, axis=1).reshape((-1, 1))

        node_norm = node_var/norm

        prob_list = np.array([np.random.choice(len(n), 1, p=n) for n in node_norm])

        if len(prob_list) == 1:
            return int(prob_list)
        else:
            return prob_list
