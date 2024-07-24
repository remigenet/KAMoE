import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pytest
import keras
from keras import random
from keras import backend
from kamoe import MoE, KAMoE


@pytest.fixture
def random_input():
    return random.normal(shape=(1000, 10, 8), dtype=backend.floatx())

def test_kamoe_lstm_return_sequences(random_input):
    moe_lstm_a = KAMoE(keras.layers.LSTM(64, return_sequences=True), n_experts=4)
    output = moe_lstm_a(random_input)
    assert output.shape == (1000, 10, 64)

def test_kamoe_lstm_no_return_sequences(random_input):
    moe_lstm_b = KAMoE(keras.layers.LSTM(64, return_sequences=False), n_experts=4)
    output = moe_lstm_b(random_input)
    assert output.shape == (1000, 64)

def test_kamoe_dense(random_input):
    moe_dense_a = KAMoE(keras.layers.Dense(64, activation='relu', use_bias=False), n_experts=4)
    output = moe_dense_a(random_input)
    assert output.shape == (1000, 10, 64)

def test_kamoe_sequential_lstm(random_input):
    base_lstm = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(16, return_sequences=True),
    ])
    moe_lstm = KAMoE(base_lstm, n_experts=4)
    output = moe_lstm(random_input)
    assert output.shape == (1000, 10, 16)

def test_kamoe_complex_sequential(random_input):
    base_complex = keras.Sequential([
        keras.layers.Dense(32),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(32),
    ])
    moe_complex = KAMoE(base_complex, n_experts=3)
    output = moe_complex(random_input)
    assert output.shape == (1000, 32)

def test_kamoe_dense_with_lstm(random_input):
    base_dense = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(32),
    ])
    moe_dense = KAMoE(base_dense, n_experts=4)
    output = moe_dense(random_input)
    assert output.shape == (1000, 32)

def test_kamoe_complex_lstm_dense(random_input):
    base_dense_2 = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(32)
    ])
    moe_dense_2 = KAMoE(base_dense_2, n_experts=4)
    output = moe_dense_2(random_input)
    assert output.shape == (1000, 32)

def test_kamoe_simple_dense_sequence(random_input):
    base_dense_3 = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32)
    ])
    moe_dense_3 = KAMoE(base_dense_3, n_experts=4)
    output = moe_dense_3(random_input)
    assert output.shape == (1000, 10, 32)

def test_kamoe_dense_with_flatten(random_input):
    base_dense_4 = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(32)
    ])
    moe_dense_4 = KAMoE(base_dense_4, n_experts=4)
    output = moe_dense_4(random_input)
    assert output.shape == (1000, 32)

def test_kamoe_dense_with_flatten_only(random_input):
    base_dense_5 = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Flatten(),
    ])
    moe_dense_5 = KAMoE(base_dense_5, n_experts=4)
    output = moe_dense_5(random_input)
    assert output.shape == (1000, 64 * 10)

def test_kamoe_custom_rnn(random_input):
    class CustomRNN(keras.layers.Layer):
        def __init__(self, units, return_sequences=True, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.return_sequences = return_sequences
            self.lstm = keras.layers.LSTM(units, return_sequences=return_sequences)
            self.dense = keras.layers.Dense(units)

        def build(self, input_shape):
            super().build(input_shape)

        def call(self, inputs):
            x = self.lstm(inputs)
            return self.dense(x)

    custom_rnn = CustomRNN(64, return_sequences=False)
    kamoe_custom = KAMoE(custom_rnn, n_experts=5)
    output = kamoe_custom(random_input)
    assert output.shape == (1000, 64)

def test_kamoe_moe(random_input):
    moe = MoE(keras.layers.Dense(64, activation='relu'), n_experts=4)
    kamoe = KAMoE(moe, n_experts=4)
    output = kamoe(random_input)
    assert output.shape == (1000, 10, 64)