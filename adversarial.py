import collections
import numpy as np
import tensorflow as tf

from universe import vectorized

from utils import plot_decision_boundary


class NoiseWrapper(vectorized.ObservationWrapper):
    def __init__(self, env):
        super(NoiseWrapper, self).__init__(env)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        adversarial_observation = self.observation(observation)
        info = self._info(observation, adversarial_observation, info)
        return adversarial_observation, reward, done, info

    def _info(self, observation_n, adversarial_observation_n, info):
        for observation, adversarial_observation, log in zip(observation_n, adversarial_observation_n, info['n']):
            diff = (observation - adversarial_observation).flatten()
            log['adversary/l2'] = np.linalg.norm(diff, ord=2) / np.sqrt(diff.shape[0])

        return info

    def _observation(self, observation_n):
        return [self._noisy(observation) for observation in observation_n]

    def _noisy(self, observation):
        return observation


class RandomNoiseWrapper(NoiseWrapper):
    def __init__(self, env, intensity=0.1):
        super(RandomNoiseWrapper, self).__init__(env)
        self.intensity = intensity

    def _noisy(self, observation):
        return observation + self.intensity * np.random.random_sample(observation.shape)


class FGSMNoiseWrapper(NoiseWrapper):
    def __init__(self, env, intensity=0.1, skip=0, reuse=False, vf=False, vf_adversarial=True, boundary=True):
        super(FGSMNoiseWrapper, self).__init__(env)
        self.intensity = intensity
        self.skip = skip
        self.reuse = reuse
        self.vf = vf
        self.vf_adversarial = vf_adversarial
        self.boundary = boundary
        self.boundary_frames = 50
        self._last_boundary_frame = 0

    def _reset(self):
        self._last_noise = None
        self._current_step = 0
        self._injects = 0
        return super(FGSMNoiseWrapper, self)._reset()

    def setup(self, policy):
        self.policy = policy
        self._policy_state = policy.get_initial_features()

        # Action probabilities given by the policy.
        y = policy.logits
        # Get the action with the highest value.
        y_true = tf.argmax(y, 1)
        # Use categorical cross-entropy as the loss function, as we want the adversarial
        # example to be as far away from possible from the true action. We assume that the
        # policy matches the actual Q function well (e.g. that argmax(y) is really the
        # optimal action to take).
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_true)
        gradient = tf.gradients(loss, policy.x)
        self.noise_op = self.intensity * tf.sign(gradient)

    def _info(self, observation_n, adversarial_observation_n, info):
        info = super(FGSMNoiseWrapper, self)._info(observation_n, adversarial_observation_n, info)

        for observation, adversarial_observation, log in zip(observation_n, adversarial_observation_n, info['n']):
            log['adversary/injects'] = self._injects

        return info

    def _get_value_function(self, observation):
        sess = tf.get_default_session()
        fetched = sess.run([self.policy.vf] + self.policy.state_out, {
            self.policy.x: [observation],
            self.policy.state_in[0]: self._policy_state[0],
            self.policy.state_in[1]: self._policy_state[1],
        })
        self._policy_state = fetched[1:]

        return fetched[0]

    def _noisy(self, observation):
        """
        Generate adversarial noise using the FGSM method for a given observation.
        """

        # Get value function.
        if self.vf:
            if self.vf_adversarial:
                vf_observation = self._last_noise if self._last_noise is not None else observation
            else:
                vf_observation = observation

            vf = self._get_value_function(vf_observation)

        if (not self.skip or self._current_step % self.skip == 0) and (not self.vf or vf > 0.5):
            # Generate noise based on the current frame.
            sess = tf.get_default_session()
            noise = sess.run(self.noise_op, {
                self.policy.x: observation.reshape([1] + list(observation.shape)),
                self.policy.state_in[0]: self._policy_state[0],
                self.policy.state_in[1]: self._policy_state[1],
            })
            self._last_noise = noise.reshape(observation.shape)
            noise = self._last_noise
            self._injects += 1

            # Visualize action decision boundary.
            if self.boundary and self._last_boundary_frame < self.boundary_frames:
                self._last_boundary_frame += 1
                attack_norm = np.linalg.norm(noise)
                b_noise = noise / attack_norm
                b_random = np.random.random_sample(b_noise.shape)
                print('frame', self._last_boundary_frame, 'attack', attack_norm, 'random', np.linalg.norm(b_random))
                b_random /= np.linalg.norm(b_random)

                def sample_policy(x):
                    samples = []
                    for sample in xrange(7):
                        fetched = self.policy.act(x, *self.policy._last_state, track_state=False)
                        samples.append(fetched[0].argmax())

                    return collections.Counter(samples).most_common()[0][0]

                def map_action(action):
                    """Map action based on semantics."""
                    # TODO: This should be based on the environment.
                    return {
                        0: 0,  # no operation
                        1: 0,  # no operation
                        2: 1,  # move up
                        3: 2,  # move down
                        4: 1,  # move up
                        5: 2,  # move down
                    }[action]

                vis_min = -2.0 * attack_norm
                vis_max = 2.0 * attack_norm
                vis_steps = 101

                b_observations = np.zeros([vis_steps, vis_steps], dtype=np.int8)
                b_samples = {}
                # Preload some coordinates to avoid images being placed there.
                sampled_images = np.array([
                    [0., 0.],
                    [attack_norm, 0.],
                ])
                limit_dist = vis_max / 10.
                limit_edge = vis_max / 4.
                for u_index, u in enumerate(np.linspace(vis_min, vis_max, vis_steps)):
                    for v_index, v in enumerate(np.linspace(vis_min, vis_max, vis_steps)):
                        b_image = observation + u * b_noise + v * b_random
                        b_observations[v_index, u_index] = map_action(sample_policy(b_image))

                        position = np.array([u, v])
                        dist = np.sum((position - sampled_images) ** 2, 1)
                        if np.min(dist) < limit_dist:
                            # Don't sample images that are too close.
                            continue
                        if position[0] < vis_min + limit_edge or position[0] > vis_max - limit_edge:
                            continue
                        if position[1] < vis_min + limit_edge or position[1] > vis_max - limit_edge:
                            continue

                        sampled_images = np.r_[sampled_images, [position]]
                        b_samples[u, v] = b_image

                # Compute the "optimal" action.
                opt_act = map_action(sample_policy(observation))

                # Plot.
                plot_decision_boundary(opt_act, self._last_boundary_frame, vis_min, vis_max, vis_steps, b_observations,
                                       adversarial_position=attack_norm, b_samples=b_samples)
        elif self.reuse:
            # Reuse last frame's noise in intermediate frames.
            noise = self._last_noise
            self._injects += 1
        else:
            # No noise in intermediate frames.
            noise = np.zeros_like(observation)

        self._current_step += 1

        return observation + noise
