import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, RNN
from tensorflow.keras.models import Sequential, clone_model
from kamoe import GRKAN, GRN
import logging

logger = logging.getLogger(__name__)

class MoE(Layer):
    def __init__(self, base_expert, n_experts=4, gating_network_activation = 'sigmoid', dropout=0., **kwargs):
        super().__init__(**kwargs)
        self.n_experts = n_experts
        self.gating_network_activation = gating_network_activation
        self.experts = [clone_model(base_expert) for _ in range(n_experts)]
        self.base_expert = self.experts[0]
        self.dropout = Dropout(dropout)
        self.input_is_sequence = None
        self.output_is_sequence = None
        self.flatten_input = None
        self.output_units = None
        self.gating_network_cls = GRN

    def build(self, input_shape):
        self.input_is_sequence = len(input_shape) == 3

        # Build one expert to determine output shape
        test_input = tf.keras.Input(shape=input_shape[1:])
        test_output = self.base_expert(test_input)
        self.output_shape = test_output.shape
        self.output_is_sequence = len(self.output_shape) == 3
        self.output_units = self.output_shape[-1]

        self.hidden_layer = Dense(self.output_units, 'relu')
        

        if self.input_is_sequence:
            if self.output_is_sequence:
                self.flatten_input = False
                weight_input_shape = input_shape
            else:
                # Check if there's an RNN layer with return_sequences=False
                if isinstance(self.base_expert, Sequential):
                    rnn_layers = [layer for layer in self.base_expert.layers if hasattr(layer, 'return_sequences')] # Using attribute return sequence over RNN type to covers wider cases
                    last_rnn_layer = next((layer for layer in reversed(rnn_layers) if not layer.return_sequences), None)
                    self.flatten_input = last_rnn_layer is None
                elif isinstance(self.base_expert, Layer):
                    if hasattr(self.base_expert, 'return_sequences'):
                        if self.base_expert.return_sequences:
                            raise ValueError('Base expert has return_sequences=True but output is not a sequence')
                        self.flatten_input = False
                    else:
                        logger.warning('Base expert does not have return_sequences attribute. Assuming it uses flattening inside')
                        self.flatten_input = True
                else:
                    logger.warning('Base expert does not have return_sequences attribute. Assuming it uses flattening inside')
                    self.flatten_input = True

                if self.flatten_input:
                    weight_input_shape = (*input_shape[:-2], input_shape[-2] * input_shape[-1])
                else:
                    weight_input_shape = (*input_shape[:-2], input_shape[-1])
        else:
            self.flatten_input = False
            weight_input_shape = input_shape

        self.hidden_layer.build(weight_input_shape)
        self.hidden_to_weight = self.gating_network_cls(self.n_experts, activation=self.gating_network_activation, dropout=self.dropout.rate)
        self.hidden_to_weight.build(self.hidden_layer.compute_output_shape(weight_input_shape))

        for expert in self.experts:
            expert.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        if self.input_is_sequence:
            if self.flatten_input:
                hidden = self.hidden_layer(tf.reshape(inputs, (tf.shape(inputs)[0], -1)))
            elif not self.output_is_sequence:
                hidden = self.hidden_layer(inputs[:, -1, :])
            else:
                hidden = self.hidden_layer(inputs)
        else:
            hidden = self.hidden_layer(inputs)

        weights = self.hidden_to_weight(hidden)
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=-1)

        if self.output_is_sequence:
            weighted_outputs = expert_outputs * weights[..., tf.newaxis, :]
        else:
            weighted_outputs = expert_outputs * weights[..., tf.newaxis, :]

        return tf.reduce_sum(weighted_outputs, axis=-1)

class KAMoE(MoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gating_network_cls = GRKAN

