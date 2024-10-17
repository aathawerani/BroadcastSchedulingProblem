import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.specs import distribution_spec
from tf_agents.utils import nest_utils
from tf_agents.networks import utils as network_utils

class ActorNet(network.DistributionNetwork):
    def __init__(self,
                 input_spec,
                 action_spec,
                 preprocessing_layers=None,
                 name=None):
        output_spec = self._get_normal_distribution_spec(action_spec)
        super(ActorNet, self).__init__(
            input_spec, (), output_spec=output_spec, name='DummyActorNet')
        self._action_spec = action_spec
        self._flat_action_spec = tf.nest.flatten(self._action_spec)[0]

        self._dummy_layers = (preprocessing_layers or []) + [
            tf.keras.layers.Dense(
                self._flat_action_spec.shape.num_elements() * 2,
                kernel_initializer=tf.compat.v1.initializers.constant([[2.0, 1.0],
                                                                       [1.0, 1.0]]),
                bias_initializer=tf.compat.v1.initializers.constant([5.0, 5.0]),
                activation=None,
            )
        ]

    def _get_normal_distribution_spec(self, sample_spec):
        input_param_shapes = tfp.distributions.Normal.param_static_shapes(
            sample_spec.shape)
        input_param_spec = tf.nest.map_structure(
            lambda tensor_shape: tensor_spec.TensorSpec(  # pylint: disable=g-long-lambda
                shape=tensor_shape,
                dtype=sample_spec.dtype),
            input_param_shapes)

        return distribution_spec.DistributionSpec(
            tfp.distributions.Normal, input_param_spec, sample_spec=sample_spec)

    def call(self, inputs, step_type=None, network_state=()):
        del step_type
        hidden_state = tf.cast(tf.nest.flatten(inputs), tf.float32)[0]

        # Calls coming from agent.train() has a time dimension. Direct loss calls
        # may not have a time dimension. It order to make BatchSquash work, we need
        # to specify the outer dimension properly.
        has_time_dim = nest_utils.get_outer_rank(inputs,
                                                 self.input_tensor_spec) == 2
        outer_rank = 2 if has_time_dim else 1
        batch_squash = network_utils.BatchSquash(outer_rank)
        hidden_state = batch_squash.flatten(hidden_state)

        for layer in self._dummy_layers:
            hidden_state = layer(hidden_state)

        actions, stdevs = tf.split(hidden_state, 2, axis=1)
        actions = batch_squash.unflatten(actions)
        stdevs = batch_squash.unflatten(stdevs)
        actions = tf.nest.pack_sequence_as(self._action_spec, [actions])
        stdevs = tf.nest.pack_sequence_as(self._action_spec, [stdevs])

        return self.output_spec.build_distribution(
            loc=actions, scale=stdevs), network_state
