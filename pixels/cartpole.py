import cv2
import numpy as np

from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv as CartPoleEnv


class CartPolePixelsEnv(CartPoleEnv):
    REPEATS = 3

    def __init__(self, *args, **kwargs):
        gravity = kwargs.pop('gravity', 9.8)
        mass_cart = kwargs.pop('mass_cart', 1.0)
        mass_pole = kwargs.pop('mass_pole', 0.1)

        super(CartPolePixelsEnv, self).__init__(*args, **kwargs)

        self.observation_space = spaces.Box(0.0, 1.0, [42, 42, self.REPEATS])

        # Update parameters.
        self.gravity = gravity
        self.masscart = mass_cart
        self.masspole = mass_pole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0

    @property
    def _height(self):
        return self.observation_space.shape[0]

    @property
    def _width(self):
        return self.observation_space.shape[1]

    def _reset(self):
        super(CartPolePixelsEnv, self)._reset()
        state = [self._render_state()] * self.REPEATS
        return np.asarray(state).transpose(1, 2, 0, 3).reshape([self._height, self._width, -1])

    def _step(self, action):
        final_state = []
        final_reward = 0
        final_done = False

        for i in xrange(self.REPEATS):
            _, reward, done, _ = super(CartPolePixelsEnv, self)._step(action)
            final_state.append(self._render_state())
            final_reward += reward

            if done:
                final_done = True
                break

        final_state = np.asarray(final_state).transpose(1, 2, 0, 3).reshape([self._height, self._width, -1])
        return final_state, final_reward, final_done, {}

    def _render_state(self):
        frame = self._render(mode='rgb_array')
        # Crop and rescale frame.
        frame = frame[:, 100:-100]
        frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (self._width, self._height))
        # Average RGB values.
        frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [self._height, self._width, 1])
        return frame
