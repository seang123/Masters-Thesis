import tensorflow as tf
import numpy as np

class Decoder(tf.keras.Model):

    def __init__(self, units=512, output_dim=256):
        super(Decoder, self).__init__()

        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, stateful=True)

        self.fc = tf.keras.layers.Dense(output_dim)

    def call(self, x):

        y, state, carry = self.lstm(x)
        y = self.fc(y)
        return y, state, carry


dec = Decoder()
dec.compile(optimizer="adam", loss="mse")

batch_size = 64
num_timesteps = 1
input_dim = 256

print("----------")
data = np.random.rand(batch_size, num_timesteps, input_dim).astype(np.float32)

y, state, carry = dec(data)

print("y", y.shape, np.mean(y))
print("state", state.shape, np.mean(state))
print("carry", carry.shape, np.mean(carry))

print("----same data------")

y, state, carry = dec(data)

print("y", y.shape, np.mean(y))
print("state", state.shape, np.mean(state))
print("carry", carry.shape, np.mean(carry))

print("----same data | reset state------")

dec.lstm.reset_states()
y, state, carry = dec(data)

print("y", y.shape, np.mean(y))
print("state", state.shape, np.mean(state))
print("carry", carry.shape, np.mean(carry))
