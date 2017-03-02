import numpy as np

from gym import spaces
from gym.envs.box2d.car_racing import CarRacing


class CarRacingPixelsEnv(CarRacing):
    def __init__(self, *args, **kwargs):
        super(CarRacingPixelsEnv, self).__init__(*args, **kwargs)

        self.action_space = spaces.Discrete(8)

    def _step(self, action):
        if action is None:
            self._last_control_state = np.array([0.0, 0.0, 0.0])
            return super(CarRacingPixelsEnv, self)._step(None)
        elif action == 0:
            self._last_control_state[0] = 1.0
        elif action == 1:
            self._last_control_state[0] = -1.0
        elif action == 2:
            self._last_control_state[0] = 0.0
        elif action == 3:
            self._last_control_state[1] = 1.0
        elif action == 4:
            self._last_control_state[1] = 0.0
        elif action == 5:
            self._last_control_state[2] = 1.0
        elif action == 6:
            self._last_control_state[2] = 0.0
        elif action == 7:
            # Do nothing.
            pass
        else:
            raise ValueError('Invalid action: {}'.format(action))

        return super(CarRacingPixelsEnv, self)._step(self._last_control_state)
