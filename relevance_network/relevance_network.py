import numpy as np

from .neural_network.network import FCLayer, Network, ReluLayer, SoftmaxCELayer


class RelevancePropagator:
    """A layer that can propagate relevance."""
    def backward_relevance(self, *args):
        raise NotImplementedError


class RelFCLayer(FCLayer, RelevancePropagator):

    def backward_relevance(self, top_rel, eps=1e-6, add_bias=False):
        """
        Propagate the relevance backwards. Unfortunately, the last layer has to be treated differently.

        :param top_rel: The relevance numbers from the next layer. If this is the last linear layer, should be a tuple
                        of the class index and the relevance value (batch x 1) from the softmax layer.
        :param eps: The epsilon for smoothing
        :param add_bias: Whether to add the bias or not. The relevance is preserved either way.
        """
        if type(top_rel) is tuple:  # Last layer
            index, value = top_rel  # index of the class and the relevance value
            weighting = self.input * self.weights[:, index, None].T
            if add_bias:
                weighting += self.bias[index]
            weighting /= (weighting.sum(axis=1, keepdims=True))
            result = weighting * value
            assert np.allclose(result.sum(axis=1), value.sum(axis=1), atol=1e-5)
            return result
        else:
            weighting = np.einsum('bx,xy->bxy', self.input, self.weights)
            if add_bias:
                weighting += self.bias
            weighting /= (weighting.sum(axis=1, keepdims=True))
            result = (weighting * top_rel[:, np.newaxis, :]).sum(axis=2)
            assert np.allclose(result.sum(axis=1), top_rel.sum(axis=1), atol=1e-5)
            return result


class RelReluLayer(ReluLayer, RelevancePropagator):

    def backward_relevance(self, top_rel, eps=1e-6, add_bias=False):
        """
        Propagate the relavance as is.
        :param top_rel: The relevance from the next layer.
        :param eps: The smoothing constant (not used)
        :param add_bias: Not used
        """
        return top_rel


class RelSoftmaxCELayer(SoftmaxCELayer, RelevancePropagator):

    def backward_relevance(self, index, eps=1e-6, add_bias=False):
        """
        :param index: The index of the class to use
        :param eps: Not used
        :param add_bias: Not used
        """
        return index, self.cache[:, index, np.newaxis] - 0.5  # subtracting 0.5 to possibly make it negative


class RelPropNetwork(Network):

    def backward_relevance(self, eps=1e-6, add_bias=False):
        """
        Perform the backward pass with relevance, assuming the forward pass has already been made.
        :param eps: The smoothing constant.
        :param add_bias: Whether to add the bias or not. The relevance is preserved either way.
        :return: A list of relevances of size (batch x input_size), one for each class
        """
        relevances = []
        n_classes = self.layers[-2].output_shape[0]
        for i in range(n_classes):
            curr_input = i
            for layer in self.layers[::-1]:
                curr_input = layer.backward_relevance(curr_input, eps, add_bias)
            relevances.append(curr_input)
        return relevances
