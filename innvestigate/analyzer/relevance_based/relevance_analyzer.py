# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals

from builtins import zip
import random
import six

###############################################################################
###############################################################################
###############################################################################

import inspect
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as kbackend
# import backend.engine.topology

from innvestigate import layers as ilayers
from innvestigate import utils as iutils
import innvestigate.backend as kutils
from ...backend import graph as kgraph, checks as kchecks
from . import relevance_rule as rrule
from . import utils as rutils
from ..network_base import AnalyzerNetworkBase
from ..reverse_base import ReverseAnalyzerBase


__all__ = [
    "BaselineLRPZ",

    "LRP",
    "LRP_RULES",

    "LRPZ",
    "LRPZIgnoreBias",

    "LRPEpsilon",
    "LRPEpsilonIgnoreBias",

    "LRPWSquare",
    "LRPFlat",

    "LRPAlphaBeta",

    "LRPAlpha2Beta1",
    "LRPAlpha2Beta1IgnoreBias",
    "LRPAlpha1Beta0",
    "LRPAlpha1Beta0IgnoreBias",
    "LRPZPlus",
    "LRPZPlusFast",

    "LRPSequentialPresetA",
    "LRPSequentialPresetB",

    "LRPSequentialPresetAFlat",
    "LRPSequentialPresetBFlat",
]


###############################################################################
###############################################################################
###############################################################################


class BaselineLRPZ(AnalyzerNetworkBase):
    """LRPZ analyzer - for testing purpose only.

    Applies the "LRP-Z" algorithm to analyze the model.
    Based on the gradient times the input formula.
    **This formula holds only for ReLU/MaxPooling networks, for which
    LRP-Z collapses into the stated formula.**

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):
        # Inside function to not break import if Keras changes.
        BASELINELRPZ_LAYERS = (
            keras.engine.topology.InputLayer,
            keras.layers.Conv1D,
            keras.layers.Conv2D,
            keras.layers.Conv2DTranspose,
            keras.layers.Conv3D,
            keras.layers.Conv3DTranspose,
            keras.layers.Cropping1D,
            keras.layers.Cropping2D,
            keras.layers.Cropping3D,
            keras.layers.SeparableConv1D,
            keras.layers.SeparableConv2D,
            keras.layers.UpSampling1D,
            keras.layers.UpSampling2D,
            keras.layers.UpSampling3D,
            keras.layers.ZeroPadding1D,
            keras.layers.ZeroPadding2D,
            keras.layers.ZeroPadding3D,
            keras.layers.Activation,
            keras.layers.ActivityRegularization,
            keras.layers.Dense,
            keras.layers.Dropout,
            keras.layers.Flatten,
            keras.layers.Lambda,
            keras.layers.Masking,
            keras.layers.Permute,
            keras.layers.RepeatVector,
            keras.layers.Reshape,
            keras.layers.SpatialDropout1D,
            keras.layers.SpatialDropout2D,
            keras.layers.SpatialDropout3D,
            keras.layers.LocallyConnected1D,
            keras.layers.LocallyConnected2D,
            keras.layers.Add,
            keras.layers.Concatenate,
            keras.layers.Dot,
            keras.layers.Maximum,
            keras.layers.Minimum,
            keras.layers.Subtract,
            keras.layers.AlphaDropout,
            keras.layers.GaussianDropout,
            keras.layers.GaussianNoise,
            keras.layers.BatchNormalization,
            keras.layers.GlobalMaxPooling1D,
            keras.layers.GlobalMaxPooling2D,
            keras.layers.GlobalMaxPooling3D,
            keras.layers.MaxPooling1D,
            keras.layers.MaxPooling2D,
            keras.layers.MaxPooling3D,
        )

        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            "BaselineLRPZ only works with  ReLU activations.",
            check_type="exception",
        )
        self._add_model_check(
            lambda layer: not isinstance(layer, BASELINELRPZ_LAYERS),
            "BaselineLRPZ only works with a predefined set of layers.",
            check_type="exception",
        )

        super(BaselineLRPZ, self).__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        tensors_to_analyze = [x for x in iutils.to_list(model.inputs)
                              if x not in stop_analysis_at_tensors]
        gradients = iutils.to_list(ilayers.Gradient()(
            tensors_to_analyze+[model.outputs[0]]))
        return [keras.layers.Multiply()([i, g])
                for i, g in zip(tensors_to_analyze, gradients)]


###############################################################################
###############################################################################
###############################################################################

# Utility list enabling name mappings via string
LRP_RULES = {
    "Z": rrule.ZRule,
    "ZIgnoreBias": rrule.ZIgnoreBiasRule,

    "Epsilon": rrule.EpsilonRule,
    "EpsilonIgnoreBias": rrule.EpsilonIgnoreBiasRule,

    "WSquare": rrule.WSquareRule,
    "Flat": rrule.FlatRule,

    "AlphaBeta": rrule.AlphaBetaRule,
    "AlphaBetaIgnoreBias": rrule.AlphaBetaIgnoreBiasRule,

    "Alpha2Beta1": rrule.Alpha2Beta1Rule,
    "Alpha2Beta1IgnoreBias": rrule.Alpha2Beta1IgnoreBiasRule,
    "Alpha1Beta0": rrule.Alpha1Beta0Rule,
    "Alpha1Beta0IgnoreBias": rrule.Alpha1Beta0IgnoreBiasRule,

    "ZPlus": rrule.ZPlusRule,
    "ZPlusFast": rrule.ZPlusFastRule,
    "Bounded": rrule.BoundedRule,
}


class EmbeddingReverseLayer(kgraph.ReverseMappingBase):
    def __init__(self, layer, state):
        # TODO: implement rule support.
        return

    def apply(self, _Xs, _Ys, Rs, _reverse_state):
        # the embedding layer outputs for an (indexed) input a vector.
        # thus, in the relevance backward pass, the embedding layer receives
        # relevances Rs corresponding to those vectors.
        # Due to the 1:1 relationship between input index and output mapping vector,
        # the relevance backward pass can be realized by pooling relevances
        # over the vector axis.

        # relevances are given shaped [batch_size, sequence_length, embedding_dims]
        pool_relevance = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=-1))
        return [pool_relevance(R) for R in Rs]


class BatchNormalizationReverseLayer(kgraph.ReverseMappingBase):
    """Special BN handler that applies the Z-Rule"""

    def __init__(self, layer, state):
        ##print("in BatchNormalizationReverseLayer.init:", layer.__class__.__name__,"-> Dedicated ReverseLayer class" ) #debug
        config = layer.get_config()

        self._center = config['center']
        self._scale = config['scale']
        self._axis = config['axis']


        self._mean = layer.moving_mean
        self._std = layer.moving_variance

        self.gamma = layer.gamma
        self.axis = layer.axis
        self._beta = layer.beta

        #TODO: implement rule support. for BatchNormalization -> [BNEpsilon, BNAlphaBeta, BNIgnore]
        #super(BatchNormalizationReverseLayer, self).__init__(layer, state)
        # how to do this:
        # super.__init__ calls select_rule and sets a self._rule class
        # check if isinstance(self_rule, EpsiloneRule), then reroute
        # to BatchNormEpsilonRule. Not pretty, but should work.

    def apply(self, Xs, Ys, Rs, reverse_state, forward = False):
        ##print("    in BatchNormalizationReverseLayer.apply:", reverse_state['layer'].__class__.__name__, '(nid: {})'.format(reverse_state['nid']))

        input_shape = [kbackend.int_shape(x) for x in Xs]
        if len(input_shape) != 1:
            # extend below lambda layers towards multiple parameters.
            raise ValueError(
                "BatchNormalizationReverseLayer expects Xs with len(Xs) = 1, but was len(Xs) = {}".format(len(Xs)))
        input_shape = input_shape[0]

        # prepare broadcasting shape for layer parameters
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self._axis[0]] = input_shape[self._axis[0]]
        broadcast_shape[0] = -1

        # reweight relevances as

        #        x * (y - beta)     R
        # Rin = ---------------- * ----
        #           x - mu          y

        # batch norm can be considered as 3 distinct layers of subtraction,
        # multiplication and then addition. The multiplicative scaling layer
        # has no effect on LRP and functions as a linear activation layer

        minus_mu = keras.layers.Lambda(lambda x: x - kbackend.reshape(self._mean, broadcast_shape))
        prepare_div = keras.layers.Lambda(
                lambda x: x + (kbackend.cast(kbackend.greater_equal(x, 0), kbackend.floatx()) * 2 - 1) * kbackend.epsilon())

        x_minus_mu = kutils.apply(minus_mu, Xs)
        if self._center:
            y_minus_beta = [ilayers.Substract()([a, b]) for a, b in zip(Ys, [self._beta])]
        else:
            y_minus_beta = Ys

        numerator = [keras.layers.Multiply()([x, ymb, r])
                    for x, ymb, r in zip(Xs, y_minus_beta, Rs)]
        denominator = [keras.layers.Multiply()([xmm, y])
                    for xmm, y in zip(x_minus_mu, Ys)]

        return [tf.math.divide_no_nan(n, prepare_div(d))
                for n, d in zip(numerator, denominator)]



class AddReverseLayer(kgraph.ReverseMappingBase):
    """Special Add layer handler that applies the Z-Rule"""

    def __init__(self, layer, state):
        ##print("in AddReverseLayer.init:", layer.__class__.__name__,"-> Dedicated ReverseLayer class" ) #debug
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             name_template="reversed_kernel_%s")

        #TODO: implement rule support.
        #super(AddReverseLayer, self).__init__(layer, state)

    def apply(self, Xs, Ys, Rs, reverse_state):
        # the outputs of the pooling operation at each location is the sum of its inputs.
        # the forward message must be known in this case, and are the inputs for each pooling thing.
        # the gradient is 1 for each output-to-input connection, which corresponds to the "weights"
        # of the layer. It should thus be sufficient to reweight the relevances  and do a gradient_wrt
        grad = ilayers.GradientWRT(len(Xs))
        # Get activations.
        Zs = kutils.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
                for a, b in zip(Rs, Zs)]

        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = kutils.to_list(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras.layers.Multiply()([a, b])
                    for a, b in zip(Xs, tmp)]




class AveragePoolingReverseLayer(kgraph.ReverseMappingBase):
    """Special AveragePooling handler that applies the Z-Rule"""

    def __init__(self, layer, state):
        ##print("in AveragePoolingRerseLayer.init:", layer.__class__.__name__,"-> Dedicated ReverseLayer class" ) #debug
        self.layer = layer
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             name_template="reversed_kernel_%s" + str(random.randint(0, 10000000)))

        #TODO: implement rule support.
        #super(AveragePoolingRerseLayer, self).__init__(layer, state)


    def apply(self, Xs, Ys, Rs, reverse_state):
        # the outputs of the pooling operation at each location is the sum of its inputs.
        # the forward message must be known in this case, and are the inputs for each pooling thing.
        # the gradient is 1 for each output-to-input connection, which corresponds to the "weights"
        # of the layer. It should thus be sufficient to reweight the relevances and and do a gradient_wrt
        grad = ilayers.GradientWRT(len(Xs))
        # Get activations.
        Zs = kutils.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
                for a, b in zip(Rs, Zs)]

        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = kutils.to_list(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras.layers.Multiply()([a, b])
                    for a, b in zip(Xs, tmp)]


class LRP(ReverseAnalyzerBase):
    """
    Base class for LRP-based model analyzers


    :param model: A Keras model.

    :param rule: A rule can be a  string or a Rule object, lists thereof or a list of conditions [(Condition, Rule), ... ]
      gradient.

    :param input_layer_rule: either a Rule object, atuple of (low, high) the min/max pixel values of the inputs
    """

    def __init__(
            self,
            model,
            *args,
            rule=None,
            input_layer_rule=None,
            until_layer_idx=None,
            until_layer_rule=None,
            bn_layer_rule=None,
            bn_layer_fuse_mode: str = "one_linear",
            **kwargs,
    ):
        super().__init__(model, *args, **kwargs)

        self._input_layer_rule = input_layer_rule
        self._until_layer_rule = until_layer_rule
        self._until_layer_idx = until_layer_idx
        self._bn_layer_rule = bn_layer_rule
        self._bn_layer_fuse_mode = bn_layer_fuse_mode

        if self._bn_layer_fuse_mode not in ["one_linear", "two_linear"]:
            raise ValueError(f"Unknown _bn_layer_fuse_mode {self._bn_layer_fuse_mode}")

        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not kchecks.is_convnet_layer(layer),
            "LRP is only tested for convolutional neural networks.",
            check_type="warning",
        )

        # check if rule was given explicitly.
        # rule can be a string, a list (of strings) or a list of conditions [(Condition, Rule), ... ] for each layer.
        if rule is None:
            raise ValueError("Need LRP rule(s).")



        if isinstance(rule, list):
            # copy refrences
            self._rule = list(rule)
        else:
            self._rule = rule
        self._input_layer_rule = input_layer_rule


        if(
           isinstance(rule, six.string_types) or
           (inspect.isclass(rule) and issubclass(rule, kgraph.ReverseMappingBase)) # NOTE: All LRP rules inherit from kgraph.ReverseMappingBase
        ):
            # the given rule is a single string or single rule implementing class
            use_conditions = True
            rules = [(lambda a, b: True, rule)]

        elif not isinstance(rule[0], tuple):
            # rule list of rule strings or classes
            use_conditions = False
            rules = list(rule)
        else:
            # rule is list of conditioned rules
            use_conditions = True
            rules = rule


        # create a BoundedRule for input layer handling from given tuple
        if self._input_layer_rule is not None:
            input_layer_rule = self._input_layer_rule
            if isinstance(input_layer_rule, tuple):
                low, high = input_layer_rule

                class BoundedProxyRule(rrule.BoundedRule):
                    def __init__(self, *args, **kwargs):
                        super(BoundedProxyRule, self).__init__(
                            *args, low=low, high=high, **kwargs)
                input_layer_rule = BoundedProxyRule


            if use_conditions is True:
                rules.insert(0,
                             (lambda layer, foo: kchecks.is_input_layer(layer),
                              input_layer_rule))

            else:
                rules.insert(0, input_layer_rule)

        self._rules_use_conditions = use_conditions
        self._rules = rules

        # FINALIZED constructor.
        super(LRP, self).__init__(model, *args, **kwargs)

    def create_rule_mapping(self, layer, reverse_state):
        ##print("in select_rule:", layer.__class__.__name__ , end='->') #debug
        rule_class = None
        if self._rules_use_conditions is True:
            for condition, rule in self._rules:
                if condition(layer, reverse_state):
                    ##print(str(rule)) #debug
                    rule_class = rule
                    break
        else:
            ##print(str(rules[0]), '(via pop)') #debug
            rule_class = self._rules.pop()

        if rule_class is None:
            raise Exception("No rule applies to layer: %s" % layer)

        if isinstance(rule_class, six.string_types):
            rule_class = LRP_RULES[rule_class]
        rule = rule_class(layer, reverse_state)

        return rule.apply

    def _create_analysis(self, *args, **kwargs):
        ####################################################################
        ### Functionality responible for backwards rule selection below ####
        ####################################################################

        # default backward hook
        self._add_conditional_reverse_mapping(
            kchecks.contains_kernel,
            self.create_rule_mapping,
            name="lrp_layer_with_kernel_mapping",
        )

        #specialized backward hooks. TODO: add ReverseLayer class handling layers Without kernel: Add and AvgPool
        self._add_conditional_reverse_mapping(
            kchecks.is_batch_normalization_layer,
            BatchNormalizationReverseLayer,
            name="lrp_batch_norm_mapping",
        )
        self._add_conditional_reverse_mapping(
            kchecks.is_average_pooling,
            AveragePoolingReverseLayer,
            name="lrp_average_pooling_mapping",
        )
        self._add_conditional_reverse_mapping(
            kchecks.is_add_layer,
            AddReverseLayer,
            name="lrp_add_layer_mapping",
        )

        self._add_conditional_reverse_mapping(
            kchecks.is_embedding_layer,
            EmbeddingReverseLayer,
            name="lrp_embedding_mapping",
        )

        # FINALIZED constructor.
        return super(LRP, self)._create_analysis(*args, **kwargs)


    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        ##print("    in _default_reverse_mapping:", reverse_state['layer'].__class__.__name__, '(nid: {})'.format(reverse_state['nid']),  end='->')
        #default_return_layers = [backend.layers.Activation]# TODO extend

        if(len(Xs) == len(Ys) and
            isinstance(reverse_state['layer'], (keras.layers.Activation,)) and
            all([kbackend.int_shape(x) == kbackend.int_shape(y) for x, y in zip(Xs, Ys)])):
            # Expect Xs and Ys to have the same shapes.
            # There is not mixing of relevances as there is kernel,
            # therefore we pass them as they are.
            ##print('return R')
            return reversed_Ys
        else:
            # This branch covers:
            # MaxPooling
            # Max
            # Flatten
            # Reshape
            # Concatenate
            # Cropping
            ##print('ilayers.GradientWRT')
            return self._gradient_reverse_mapping(
                    Xs, Ys, reversed_Ys, reverse_state)

    ########################################
    ### End of Rule Selection Business. ####
    ########################################


    def _get_state(self):
        state = super(LRP, self)._get_state()
        state.update({"rule": self._rule})
        state.update({"input_layer_rule": self._input_layer_rule})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        rule = state.pop("rule")
        input_layer_rule = state.pop("input_layer_rule")
        kwargs = super(LRP, clazz)._state_to_kwargs(state)
        kwargs.update({"rule": rule,
                       "input_layer_rule": input_layer_rule})
        return kwargs


###############################################################################
# ANALYZER CLASSES AND PRESETS ################################################
###############################################################################


class _LRPFixedParams(LRP):

    @classmethod
    def _state_to_kwargs(clazz, state):
        kwargs = super(_LRPFixedParams, clazz)._state_to_kwargs(state)
        del kwargs["rule"]
        return kwargs


class LRPZ(_LRPFixedParams):
    """LRP-analyzer that uses the LRP-Z rule"""
    
    def __init__(self, model, *args, **kwargs):
        super(LRPZ, self).__init__(model, *args, rule="Z", **kwargs)


class LRPZIgnoreBias(_LRPFixedParams):
    """LRP-analyzer that uses the LRP-Z-ignore-bias rule"""

    def __init__(self, model, *args, **kwargs):
        super(LRPZIgnoreBias, self).__init__(model, *args,
                                             rule="ZIgnoreBias", **kwargs)



class LRPEpsilon(_LRPFixedParams):
    """LRP-analyzer that uses the LRP-Epsilon rule"""

    def __init__(self, model, epsilon=1e-7, bias=True, *args, **kwargs):

        class EpsilonProxyRule(rrule.EpsilonRule):
            """
            Dummy class inheriting from EpsilonRule
            for passing along the chosen parameters from
            the LRP analyzer class to the decopmosition rules.
            """
            def __init__(self, *args, **kwargs):
                super(EpsilonProxyRule, self).__init__(*args,
                                                       epsilon=epsilon,
                                                       bias=bias,
                                                       **kwargs)

        super(LRPEpsilon, self).__init__(model, *args,
                                         rule=EpsilonProxyRule,
                                         **kwargs)


class LRPEpsilonIgnoreBias(LRPEpsilon):
    """LRP-analyzer that uses the LRP-Epsilon-ignore-bias rule"""

    def __init__(self, model, epsilon=1e-7, *args, **kwargs):
        super(LRPEpsilonIgnoreBias, self).__init__(model, *args,
                                                   epsilon=epsilon,
                                                   bias=False,
                                                   **kwargs)


class LRPWSquare(_LRPFixedParams):
    """LRP-analyzer that uses the DeepTaylor W**2 rule"""

    def __init__(self, model, *args, **kwargs):
        super(LRPWSquare, self).__init__(model, *args,
                                         rule="WSquare", **kwargs)


class LRPFlat(_LRPFixedParams):
    """LRP-analyzer that uses the LRP-Flat rule"""

    def __init__(self, model, *args, **kwargs):
        super(LRPFlat, self).__init__(model, *args,
                                      rule="Flat", **kwargs)


class LRPAlphaBeta(LRP):
    """ Base class for LRP AlphaBeta"""

    def __init__(self, model, alpha=None, beta=None, bias=True, *args, **kwargs):
        alpha, beta = rutils.assert_infer_lrp_alpha_beta_param(alpha, beta, self)
        self._alpha = alpha
        self._beta = beta
        self._bias = bias

        class AlphaBetaProxyRule(rrule.AlphaBetaRule):
            """
            Dummy class inheriting from AlphaBetaRule
            for the purpose of passing along the chosen parameters from
            the LRP analyzer class to the decopmosition rules.
            """
            def __init__(self, *args, **kwargs):
                super(AlphaBetaProxyRule, self).__init__(*args,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias,
                                                              **kwargs)

        super(LRPAlphaBeta, self).__init__(model, *args,
                                           rule=AlphaBetaProxyRule,
                                           **kwargs)

    def _get_state(self):
        state = super(LRPAlphaBeta, self)._get_state()
        del state["rule"]
        state.update({"alpha": self._alpha})
        state.update({"beta": self._beta})
        state.update({"bias": self._bias})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        alpha = state.pop("alpha")
        beta = state.pop("beta")
        bias = state.pop("bias")
        state["rule"] = None
        kwargs = super(LRPAlphaBeta, clazz)._state_to_kwargs(state)
        del kwargs["rule"]
        kwargs.update({"alpha": alpha,
                       "beta": beta,
                       "bias": bias})
        return kwargs






class _LRPAlphaBetaFixedParams(LRPAlphaBeta):

    @classmethod
    def _state_to_kwargs(clazz, state):
        kwargs = super(_LRPAlphaBetaFixedParams, clazz)._state_to_kwargs(state)
        del kwargs["alpha"]
        del kwargs["beta"]
        del kwargs["bias"]
        return kwargs


class LRPAlpha2Beta1(_LRPAlphaBetaFixedParams):
    """LRP-analyzer that uses the LRP-alpha-beta rule with a=2,b=1"""

    def __init__(self, model, *args, **kwargs):
        super(LRPAlpha2Beta1, self).__init__(model, *args,
                                             alpha=2,
                                             beta=1,
                                             bias=True,
                                             **kwargs)


class LRPAlpha2Beta1IgnoreBias(_LRPAlphaBetaFixedParams):
    """LRP-analyzer that uses the LRP-alpha-beta-ignbias rule with a=2,b=1"""

    def __init__(self, model, *args, **kwargs):
        super(LRPAlpha2Beta1IgnoreBias, self).__init__(model, *args,
                                                       alpha=2,
                                                       beta=1,
                                                       bias=False,
                                                       **kwargs)


class LRPAlpha1Beta0(_LRPAlphaBetaFixedParams):
    """LRP-analyzer that uses the LRP-alpha-beta rule with a=1,b=0"""

    def __init__(self, model, *args, **kwargs):
        super(LRPAlpha1Beta0, self).__init__(model, *args,
                                             alpha=1,
                                             beta=0,
                                             bias=True,
                                             **kwargs)


class LRPAlpha1Beta0IgnoreBias(_LRPAlphaBetaFixedParams):
    """LRP-analyzer that uses the LRP-alpha-beta-ignbias rule with a=1,b=0"""

    def __init__(self, model, *args, **kwargs):
        super(LRPAlpha1Beta0IgnoreBias, self).__init__(model, *args,
                                                       alpha=1,
                                                       beta=0,
                                                       bias=False,
                                                       **kwargs)

class LRPZPlus(LRPAlpha1Beta0IgnoreBias):
    """LRP-analyzer that uses the LRP-alpha-beta rule with a=1,b=0"""
    #TODO: assert that layer inputs are always >= 0
    def __init__(self, model, *args, **kwargs):
        super(LRPZPlus, self).__init__(model, *args, **kwargs)


class LRPZPlusFast(_LRPFixedParams):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0 and assumes inputs x >= 0.
    """
    #TODO: assert that layer inputs are always >= 0
    def __init__(self, model, *args, **kwargs):
        super(LRPZPlusFast, self).__init__(model, *args,
                                       rule="ZPlusFast", **kwargs)


class LRPSequentialPresetA(_LRPFixedParams): #for the lack of a better name
    """Special LRP-configuration for ConvNets"""

    def __init__(self, model, epsilon=1e-1, *args, **kwargs):

        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            #TODO: fix. specify. extend.
            ("LRPSequentialPresetA is not advised "
             "for networks with non-ReLU activations."),
            check_type="warning",
        )

        class EpsilonProxyRule(rrule.EpsilonRule):
            def __init__(self, *args, **kwargs):
                super(EpsilonProxyRule, self).__init__(*args,
                                                       epsilon=epsilon,
                                                       bias=True,
                                                       **kwargs)


        conditional_rules = [(kchecks.is_dense_layer, EpsilonProxyRule),
                             (kchecks.is_conv_layer, rrule.Alpha1Beta0Rule)
                            ]

        super(LRPSequentialPresetA, self).__init__(model,
                                            *args,
                                            rule = conditional_rules,
                                            **kwargs )


class LRPSequentialPresetB(_LRPFixedParams):
    """Special LRP-configuration for ConvNets"""

    def __init__(self, model, epsilon=1e-1, *args, **kwargs):
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            #TODO: fix. specify. extend.
            ("LRPSequentialPresetB is not advised "
             "for networks with non-ReLU activations."),
            check_type="warning",
        )

        class EpsilonProxyRule(rrule.EpsilonRule):
            def __init__(self, *args, **kwargs):
                super(EpsilonProxyRule, self).__init__(*args,
                                                       epsilon=epsilon,
                                                       bias=True,
                                                       **kwargs)


        conditional_rules = [(kchecks.is_dense_layer, EpsilonProxyRule),
                             (kchecks.is_conv_layer, rrule.Alpha2Beta1Rule)
                            ]
        super(LRPSequentialPresetB, self).__init__(model,
                                            *args,
                                            rule = conditional_rules,
                                            **kwargs )





#TODO: allow to pass input layer identification by index or id.
class LRPSequentialPresetAFlat(LRPSequentialPresetA):
    """Special LRP-configuration for ConvNets"""

    def __init__(self, model, *args, **kwargs):
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            # TODO: fix. specify. extend.
            ("LRPSequentialPresetB is not advised "
             "for networks with non-ReLU activations."),
            check_type="warning",
        )

        super(LRPSequentialPresetAFlat, self).__init__(model,
                                                *args,
                                                input_layer_rule=rrule.FlatRule,
                                                **kwargs)



#TODO: allow to pass input layer identification by index or id.
class LRPSequentialPresetBFlat(LRPSequentialPresetB):
    """Special LRP-configuration for ConvNets"""

    def __init__(self, model, *args, **kwargs):
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            # TODO: fix. specify. extend.
            ("LRPSequentialPresetB is not advised "
             "for networks with non-ReLU activations."),
            check_type="warning",
        )
        super(LRPSequentialPresetBFlat, self).__init__(model,
                                                *args,
                                                input_layer_rule="Flat",
                                                **kwargs)



class LRPSequentialPresetBFlatUntilIdx(LRPSequentialPresetBFlat):
    """
    Special LRP-configuration for ConvNets.
    Allows to perform LRP_flat from (including) layer until_layer_idx down until
    the input layer. Weightless layers are ignored when counting the index for now.
    """

    def __init__(self, model, *args, until_layer_idx=None, **kwargs):
        super(LRPSequentialPresetBFlatUntilIdx, self).__init__(model,
                                                       *args,
                                                               until_layer_idx=until_layer_idx,
                                                               until_layer_rule=rrule.FlatRule,
                                                               **kwargs,
                                                               )
