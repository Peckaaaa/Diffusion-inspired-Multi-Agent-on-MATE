"""Tests for the DIMA x MATE research layer (brief section 33).

Neither upstream repository ships tests or a test runner, so there is no
convention to follow and no dependency to add: these use the standard library's
``unittest``.

    python -m unittest tests.test_research -v
    RESEARCH_SLOW_TESTS=1 python -m unittest tests.test_research -v   # + the full pipeline

The slow test builds DIMA's 14M-parameter denoiser, so it is opt-in.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research  # noqa: E402,F401 - installs sys.path + compat shims

from research.env_adapter import MATEEnv, denormalize_observation  # noqa: E402
from research.planners import (  # noqa: E402
    NEEDS_ACTOR,
    PLANNER_REGISTRY,
    DIMAActorPlanner,
    build_planner,
)
from research.rollout import run_episode, to_dima_rollout  # noqa: E402
from research.views import ObservationLayout, SceneView  # noqa: E402
from research.world_model import AlphaOracleWorldModel, History, OracleWorldModel  # noqa: E402


SLOW = os.environ.get('RESEARCH_SLOW_TESTS') == '1'


def make_env(**kwargs) -> MATEEnv:
    defaults = dict(scenario='MATE-4v2-9', seed=0, discrete_levels=5, max_episode_steps=40)
    defaults.update(kwargs)
    return MATEEnv(**defaults)


class TestCompat(unittest.TestCase):
    def test_shims_report_what_they_did(self):
        self.assertIn('numpy.bool8', research.COMPAT_SHIMS)
        self.assertIn('gym.seeding.np_random', research.COMPAT_SHIMS)

    def test_mate_imports_as_the_real_package(self):
        import mate

        self.assertTrue(hasattr(mate, 'ASSETS_DIR'))
        self.assertEqual(mate.__version__, '0.1.0')


class TestEnvAdapter(unittest.TestCase):
    """Brief sections 4, 8, 9."""

    @classmethod
    def setUpClass(cls):
        cls.env = make_env()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_metadata_comes_from_mate(self):
        unwrapped = self.env.unwrapped
        self.assertEqual(self.env.n_agents, unwrapped.num_cameras)
        self.assertEqual(self.env.n_obs, unwrapped.camera_observation_space.shape[0])
        self.assertEqual(self.env.state_dim, unwrapped.state_space.shape[0])
        self.assertEqual(self.env.n_actions, self.env.discrete_levels**2)
        self.assertEqual(self.env.max_time_steps, unwrapped.max_episode_steps)

    def test_reset_and_step_shapes(self):
        observation = self.env.reset(seed=0)
        self.assertEqual(observation.obs_raw.shape, (self.env.n_agents, self.env.n_obs))
        self.assertEqual(observation.obs.shape, (self.env.n_agents, self.env.n_obs))
        self.assertEqual(observation.state_raw.shape, (self.env.state_dim,))

        action = np.zeros(self.env.n_agents, dtype=np.int64)
        observation, reward, done, info = self.env.step(action)
        self.assertEqual(reward.shape, (self.env.n_agents,))
        self.assertIsInstance(done, bool)
        self.assertIn('coverage_rate', info)

    def test_rejects_malformed_actions(self):
        self.env.reset(seed=0)
        with self.assertRaises(ValueError):
            self.env.step(np.zeros(self.env.n_agents + 1, dtype=np.int64))
        with self.assertRaises(ValueError):
            self.env.step(np.full(self.env.n_agents, self.env.n_actions, dtype=np.int64))

    def test_rescale_round_trip_is_exact(self):
        observation = self.env.reset(seed=1)
        for _ in range(5):
            observation, *_ = self.env.step(np.full(self.env.n_agents, 12, dtype=np.int64))
        self.assertLess(np.abs(self.env.unrescale_obs(observation.obs) - observation.obs_raw).max(), 1e-8)
        self.assertLess(
            np.abs(self.env.unrescale_state(observation.state) - observation.state_raw).max(), 1e-8
        )

    def test_denormalize_matches_mate_forward_pass(self):
        space = self.env.camera_observation_space
        raw = self.env.reset(seed=2).obs_raw
        self.assertLess(np.abs(denormalize_observation(self.env.rescale_obs(raw), space) - raw).max(), 1e-8)

    def test_fork_does_not_disturb_the_live_episode(self):
        self.env.reset(seed=3)
        for _ in range(3):
            self.env.step(np.full(self.env.n_agents, 12, dtype=np.int64))
        step_before = self.env.episode_step
        fork = self.env.fork()
        fork.step(np.zeros(self.env.n_agents, dtype=np.int64))
        self.assertEqual(self.env.episode_step, step_before)
        self.assertEqual(fork.episode_step, step_before + 1)
        fork.close()


class TestActionConversion(unittest.TestCase):
    """Brief section 8: the action representation must be lossless."""

    def test_every_discrete_action_round_trips(self):
        env = make_env()
        try:
            for index in range(env.n_actions):
                actions = np.full(env.n_agents, index, dtype=np.int64)
                recovered = env.action_from_continuous(env.action_to_continuous(actions))
                np.testing.assert_array_equal(recovered, actions)
        finally:
            env.close()

    def test_centre_action_is_the_no_op(self):
        env = make_env()
        try:
            centre = env.n_actions // 2
            np.testing.assert_allclose(env.action_to_continuous([centre])[0], [0.0, 0.0])
        finally:
            env.close()


class TestSceneView(unittest.TestCase):
    """Brief section 13: the standardized representation must decode MATE correctly."""

    def test_coverage_estimate_matches_mate(self):
        env = make_env(max_episode_steps=60, seed=7)
        try:
            layout = ObservationLayout.from_env_metadata(env.metadata())
            observation = env.reset(seed=7)
            rng = np.random.default_rng(0)
            for _ in range(40):
                observation, _, done, info = env.step(rng.integers(0, env.n_actions, env.n_agents))
                view = SceneView.from_joint_observation(observation.obs_raw, layout)
                self.assertAlmostEqual(view.coverage_estimate(), info['coverage_rate'], places=9)
                if done:
                    break
        finally:
            env.close()

    def test_soft_coverage_is_non_negative_exactly_when_covered(self):
        env = make_env(max_episode_steps=60, seed=11)
        try:
            layout = ObservationLayout.from_env_metadata(env.metadata())
            observation = env.reset(seed=11)
            rng = np.random.default_rng(1)
            for _ in range(30):
                observation, *_ = env.step(rng.integers(0, env.n_actions, env.n_agents))
                view = SceneView.from_joint_observation(observation.obs_raw, layout)
                np.testing.assert_array_equal(
                    view.margin_matrix() >= 0.0, view.tracking_matrix()
                )
        finally:
            env.close()


class TestPlanners(unittest.TestCase):
    """Brief sections 14, 16."""

    def test_every_reactive_planner_returns_a_legal_joint_action(self):
        env = make_env(max_episode_steps=20)
        try:
            for spec, cls in PLANNER_REGISTRY.items():
                # Skipped for opposite reasons: one plans with a world model, the
                # other acts with a policy network. Neither is buildable from an
                # environment alone, which is what this test covers.
                if cls.USES_WORLD_MODEL or spec in NEEDS_ACTOR:
                    continue
                planner = build_planner(spec, env, seed=0)
                result = run_episode(env, planner, seed=0, max_steps=20)
                self.assertGreater(len(result), 0, spec)
                for transition in result.transitions:
                    self.assertEqual(transition.action.shape, (env.n_agents,), spec)
                    self.assertTrue(
                        ((0 <= transition.action) & (transition.action < env.n_actions)).all(), spec
                    )
        finally:
            env.close()

    def test_actor_planner_acts_with_the_policy_network(self):
        """The learned policy has to be usable as a planner, or online training
        cannot close its loop."""

        from networks.dreamer.action import StochasticPolicy

        from research.config import build_learner_config

        env = make_env(max_episode_steps=20)
        try:
            config = build_learner_config(env, seed=0, device='cpu')
            actor = StochasticPolicy(
                config.IN_DIM,
                config.ACTION_SIZE,
                config.ACTION_HIDDEN,
                config.ACTION_LAYERS,
                continuous_action=config.CONTINUOUS_ACTION,
                continuous_action_space=config.ACTION_SPACE,
                policy_class=config.policy_class,
            )
            planner = DIMAActorPlanner(env, actor, seed=0)
            result = run_episode(env, planner, seed=0, max_steps=20)

            self.assertGreater(len(result), 0)
            for transition in result.transitions:
                self.assertEqual(transition.action.shape, (env.n_agents,))
                self.assertTrue(((0 <= transition.action) & (transition.action < env.n_actions)).all())

            # Whatever the actor produces has to be the same dict the learner is
            # fed offline, or the two training paths diverge silently.
            scripted = to_dima_rollout(
                run_episode(env, build_planner('random', env, seed=0), seed=0, max_steps=20),
                env.n_actions,
            )
            online = to_dima_rollout(result, env.n_actions)
            self.assertEqual(set(online), set(scripted))
            for key in scripted:
                self.assertEqual(online[key].shape[1:], scripted[key].shape[1:], key)
        finally:
            env.close()

    def test_actor_planner_without_an_actor_is_refused(self):
        env = make_env(max_episode_steps=10)
        try:
            with self.assertRaises(ValueError):
                build_planner('dima_actor', env, seed=0)
        finally:
            env.close()

    def test_entry_point_specs_resolve(self):
        env = make_env(max_episode_steps=10)
        try:
            planner = build_planner('research.planners:RandomPlanner', env, seed=0)
            self.assertEqual(type(planner).__name__, 'RandomPlanner')
        finally:
            env.close()

    def test_world_model_planner_without_a_model_is_refused(self):
        env = make_env(max_episode_steps=10)
        try:
            with self.assertRaises(ValueError):
                build_planner('predictive_greedy', env, seed=0)
        finally:
            env.close()

    def test_privileged_state_is_gated(self):
        from research.rollout import _make_context

        env = make_env(max_episode_steps=10)
        try:
            layout = ObservationLayout.from_env_metadata(env.metadata())
            observation = env.reset(seed=0)
            planner = build_planner('reactive_greedy', env, seed=0)
            history = History(length=1)
            history.seed(observation, env.n_agents, env.n_actions // 2)
            context = _make_context(env, planner, layout, observation, history, episode=0, step=0)
            self.assertIsNone(context.privileged)
        finally:
            env.close()


class TestDIMAInterop(unittest.TestCase):
    """Brief section 10: MATE trajectories must fit DIMA's existing abstractions."""

    def test_rollout_is_accepted_by_dima_dataset(self):
        import torch

        from dataset import MultiAgentEpisodesDataset
        from episode import MamujocoEpisode

        env = make_env(max_episode_steps=40)
        try:
            planner = build_planner('random', env, seed=0)
            result = run_episode(env, planner, seed=0, max_steps=40)
            rollout = to_dima_rollout(result, env.n_actions)

            steps = len(result)
            self.assertEqual(rollout['observation'].shape, (steps, env.n_agents, env.n_obs))
            self.assertEqual(rollout['shared_obs'].shape, (steps, env.n_agents, env.state_dim))
            self.assertEqual(rollout['action'].shape, (steps, env.n_agents, env.n_actions))
            self.assertTrue((rollout['action'].sum(-1) == 1).all())
            self.assertEqual(rollout['last'][-1].sum(), env.n_agents)

            episode = MamujocoEpisode(
                observation=torch.FloatTensor(rollout['observation']),
                shared_obs=torch.FloatTensor(rollout['shared_obs']),
                next_shared_obs=torch.FloatTensor(rollout['next_shared_obs']),
                action=torch.FloatTensor(rollout['action']),
                reward=torch.FloatTensor(rollout['reward']),
                done=torch.FloatTensor(rollout['done']),
                filled=torch.ones(steps, dtype=torch.bool),
            )
            dataset = MultiAgentEpisodesDataset(
                max_ram_usage='1G', name='t', capacity=1024, diffusion_seq_len=6, condition_steps=3
            )
            dataset.add_episode(episode)
            batch = dataset.sample_batch(batch_num_samples=2, sequence_length=6)
            self.assertEqual(
                tuple(batch.shared_obs.shape), (2, 6, env.n_agents, env.state_dim)
            )
        finally:
            env.close()

    def test_config_dimensions_are_detected_not_declared(self):
        from research.config import build_learner_config

        env = make_env()
        try:
            config = build_learner_config(env, seed=0, horizon=5)
            self.assertEqual(config.IN_DIM, env.n_obs)
            self.assertEqual(config.STATE_DIM, env.state_dim)
            self.assertEqual(config.ACTION_SIZE, env.n_actions)
            self.assertEqual(config.NUM_AGENTS, env.n_agents)
            self.assertFalse(config.CONTINUOUS_ACTION)
            # The default is still one denoising step per agent, but under joint
            # action conditioning that is a default, not a constraint --
            # sample_agent_order is not reached at all.
            self.assertEqual(config.diffusion_sampler_cfg.num_steps_denoising, config.NUM_AGENTS)
            self.assertEqual(config.denoiser_cfg.inner_model.action_cond, 'joint')
            self.assertEqual(config.rew_end_model_type, 'transformer')
        finally:
            env.close()


class TestBufferSizing(unittest.TestCase):
    """The batch sizes and the minimum buffer are one constraint, not two."""

    def test_minimum_buffer_steps_follows_the_batch_sizes(self):
        from research.train_wm import minimum_buffer_steps

        # DIMA's own numbers, which this bound was originally written against.
        self.assertEqual(
            minimum_buffer_steps(5, state_decoder_batch_size=256, rew_end_batch_size=128), 644
        )
        # Raising a batch size raises the buffer it needs; a bound that ignored
        # this is what makes DreamerMemory raise a bare 'Not enough data in buffer'
        # deep inside training.
        self.assertGreater(
            minimum_buffer_steps(5, state_decoder_batch_size=1024, rew_end_batch_size=512),
            minimum_buffer_steps(5, state_decoder_batch_size=256, rew_end_batch_size=128),
        )
        # The state decoder draws sl=1, so it dominates only at short horizons.
        self.assertEqual(
            minimum_buffer_steps(1, state_decoder_batch_size=1024, rew_end_batch_size=64), 1024
        )


class TestWorldModelInterface(unittest.TestCase):
    """Brief sections 12, 28, 29."""

    def _history(self, env, world_model):
        observation = env.reset(seed=0)
        history = History(length=world_model.conditioning_steps)
        history.seed(observation, env.n_agents, env.n_actions // 2)
        return history

    def test_oracle_prediction_shapes(self):
        env = make_env(max_episode_steps=40)
        try:
            oracle = OracleWorldModel(env)
            history = self._history(env, oracle)
            candidates = np.array([[0, 0, 0, 0], [24, 24, 24, 24], [12, 12, 12, 12]])
            prediction = oracle.predict(history, candidates, horizon=2)
            self.assertEqual(
                prediction.observations.shape, (3, 2, env.n_agents, env.n_obs)
            )
            self.assertEqual(prediction.states.shape, (3, 2, env.state_dim))
            self.assertEqual(prediction.rewards.shape, (3, 2))
            self.assertTrue(np.isfinite(prediction.observations).all())
        finally:
            env.close()

    def test_oracle_distinguishes_actions(self):
        env = make_env(max_episode_steps=40)
        try:
            oracle = OracleWorldModel(env)
            history = self._history(env, oracle)
            candidates = np.array([[0, 12, 12, 12], [24, 12, 12, 12]])
            prediction = oracle.predict(history, candidates, horizon=1)
            spread = np.abs(prediction.observations[0] - prediction.observations[1]).max()
            self.assertGreater(spread, 0.0, 'the oracle must react to a changed action')
        finally:
            env.close()

    def test_alpha_oracle_interpolates(self):
        env = make_env(max_episode_steps=40)
        try:
            candidates = np.array([[0, 12, 12, 12], [24, 12, 12, 12]])

            zero = AlphaOracleWorldModel(env, alpha=0.0, seed=0)
            history = self._history(env, zero)
            prediction = zero.predict(history, candidates, horizon=1)
            np.testing.assert_allclose(
                prediction.observations[0], prediction.observations[1], atol=1e-9
            )

            one = AlphaOracleWorldModel(env, alpha=1.0, seed=0)
            history = self._history(env, one)
            attenuated = one.predict(history, candidates, horizon=1)
            spread = np.abs(attenuated.observations[0] - attenuated.observations[1]).max()
            self.assertGreater(spread, 0.0)
        finally:
            env.close()

    def test_action_shape_validation(self):
        env = make_env(max_episode_steps=20)
        try:
            oracle = OracleWorldModel(env)
            history = self._history(env, oracle)
            with self.assertRaises(ValueError):
                oracle.predict(history, np.zeros((2, 3), dtype=np.int64), horizon=1)
            with self.assertRaises(ValueError):
                oracle.predict(history, np.zeros((2, 5, 4), dtype=np.int64), horizon=2)
        finally:
            env.close()


class TestHistoryAlignment(unittest.TestCase):
    """The state/action alignment the whole world-model path depends on.

    ``actions[-1]`` must be the action that produced ``states[-1]``; the action
    paired with the *current* state is the candidate a planner is choosing.  An
    off-by-one here is silent -- the model still returns plausible states, just
    conditioned on the wrong action -- so it is pinned by a test.
    """

    def test_push_records_action_and_resulting_state_together(self):
        env = make_env(max_episode_steps=20)
        try:
            history = History(length=3)
            observation = env.reset(seed=0)
            history.seed(observation, env.n_agents, env.n_actions // 2)

            applied = []
            for step in range(4):
                action = np.full(env.n_agents, step + 1, dtype=np.int64)
                observation, *_ = env.step(action)
                history.push(observation, action)
                applied.append(action)

                np.testing.assert_array_equal(history.action_array()[-1], action)
                np.testing.assert_allclose(history.state_array()[-1], observation.state)
        finally:
            env.close()

    def test_conditioning_actions_excludes_the_candidate_slot(self):
        env = make_env(max_episode_steps=20)
        try:
            history = History(length=3)
            observation = env.reset(seed=0)
            history.seed(observation, env.n_agents, env.n_actions // 2)
            for step in range(4):
                observation, *_ = env.step(np.full(env.n_agents, step + 1, dtype=np.int64))
                history.push(observation, np.full(env.n_agents, step + 1, dtype=np.int64))

            conditioning = history.conditioning_actions(3)
            self.assertEqual(conditioning.shape, (2, env.n_agents))
            # actions were 1,2,3,4; the window of length 3 conditions on the two
            # that precede the candidate, i.e. 3 and 4.
            np.testing.assert_array_equal(conditioning[:, 0], [3, 4])

            self.assertEqual(history.conditioning_actions(1).shape[0], 0)
        finally:
            env.close()


class TestTorchSetup(unittest.TestCase):
    """The GPU/stability settings applied before any DIMA module is built."""

    def test_anomaly_detection_is_off_by_default(self):
        import torch

        from research.config import configure_torch

        torch.autograd.set_detect_anomaly(True)
        applied = configure_torch('cpu')
        self.assertFalse(applied['detect_anomaly'])
        # A grad-requiring op is the only way to observe the global flag.
        x = torch.zeros(1, requires_grad=True)
        (x * 2).sum().backward()
        self.assertIn('torch_threads', applied)

    def test_seeding_is_recorded_and_deterministic(self):
        import torch

        from research.config import configure_torch

        applied = configure_torch('cpu', seed=7)
        self.assertEqual(applied['seed'], 7)
        first = torch.randn(4)
        configure_torch('cpu', seed=7)
        np.testing.assert_allclose(first.numpy(), torch.randn(4).numpy())


class TestValidation(unittest.TestCase):
    """Held-out validation, the signal the server run is watched with."""

    def _rollout(self, env, steps=30):
        planner = build_planner('random', env, seed=0)
        result = run_episode(env, planner, seed=0, max_steps=steps)
        return to_dima_rollout(result, env.n_actions), result

    def test_validation_episode_reconstructs_the_successor_observation(self):
        from research.validation import ValidationEpisode

        env = make_env(max_episode_steps=30)
        try:
            rollout, result = self._rollout(env, 30)
            episode = ValidationEpisode.from_rollout(rollout, env)

            # One shorter: the last step has no stored successor.
            self.assertEqual(len(episode.transitions), len(result.transitions) - 1)
            for t, step in enumerate(episode.transitions):
                # The dataset stores float32, so a coordinate of ~1000 round-trips
                # to about 1e-5; compare at float32 precision, not float64.
                np.testing.assert_allclose(
                    step.next_obs_raw, result.transitions[t].next_obs_raw, rtol=1e-5, atol=1e-3
                )
                np.testing.assert_array_equal(step.action, result.transitions[t].action)
        finally:
            env.close()

    def test_validate_reports_the_metrics_training_is_watched_on(self):
        from research.validation import ValidationEpisode, format_trend, validate

        env = make_env(max_episode_steps=30)
        try:
            episodes = [ValidationEpisode.from_rollout(self._rollout(env, 30)[0], env)]
            layout = ObservationLayout.from_env_metadata(env.metadata())
            row = validate(
                OracleWorldModel(env),
                episodes,
                layout,
                num_agents=env.n_agents,
                num_actions=env.n_actions,
                states_per_episode=2,
                sensitivity_samples=2,
                sensitivity_actions=3,
            )
            for key in ('mae_h1', 'rmse_h1', 'ade_h1', 'sensitivity_ratio'):
                self.assertIn(key, row)
            # The oracle is the real environment, so it must be far from action-blind.
            self.assertGreater(row['sensitivity_ratio'], 1.0)

            line = format_trend([row], ('ade_h1', 'sensitivity_ratio'))
            self.assertIn('sensitivity_ratio=', line)
            self.assertIn('N/A', format_trend([{'ade_h1': None}], ('ade_h1',)))
        finally:
            env.close()

    def test_validation_scores_the_same_states_every_call(self):
        """The trend must reflect the model, not which states got sampled."""

        from research.validation import ValidationEpisode, validate

        env = make_env(max_episode_steps=40)
        try:
            episodes = [
                ValidationEpisode.from_rollout(self._rollout(env, 40)[0], env) for _ in range(2)
            ]
            layout = ObservationLayout.from_env_metadata(env.metadata())
            kwargs = dict(
                num_agents=env.n_agents,
                num_actions=env.n_actions,
                states_per_episode=3,
                sensitivity_samples=2,
                sensitivity_actions=3,
                sensitivity_states=2,
            )
            model = OracleWorldModel(env)

            first = validate(model, episodes, layout, seed=0, **kwargs)
            # Disturb the global RNG: a correct implementation is unaffected.
            np.random.default_rng(123).random(50)
            second = validate(model, episodes, layout, seed=0, **kwargs)
            self.assertEqual(first, second)

            # ...and the seed is genuinely what selects the states. `mae_h1` is
            # always defined; `ade_h1` is None when no target was sighted in the
            # sampled ground-truth states, which is common in short random rollouts.
            other = validate(model, episodes, layout, seed=99, **kwargs)
            self.assertNotEqual(first['mae_h1'], other['mae_h1'])
        finally:
            env.close()

    def test_trend_arrows_follow_the_direction_of_change(self):
        from research.validation import format_trend

        rising = format_trend([{'x': 1.0}, {'x': 2.0}], ('x',))
        falling = format_trend([{'x': 2.0}, {'x': 1.0}], ('x',))
        flat = format_trend([{'x': 1.0}, {'x': 1.0}], ('x',))
        self.assertIn('↑', rising)
        self.assertIn('↓', falling)
        self.assertIn('→', flat)


class TestOraclePlannerBeatsRandom(unittest.TestCase):
    """The planner interface works if a perfect model makes it act better."""

    def test_oracle_planner_outperforms_random(self):
        coverages = {}
        for spec, world_model_factory in (
            ('random', None),
            ('oracle', OracleWorldModel),
        ):
            scores = []
            for episode in range(3):
                env = make_env(max_episode_steps=60, seed=episode)
                world_model = world_model_factory(env) if world_model_factory else None
                planner = build_planner(spec, env, world_model=world_model, seed=episode)
                result = run_episode(
                    env,
                    planner,
                    world_model=world_model,
                    seed=episode,
                    max_steps=60,
                    reference_prediction=False,
                )
                scores.append(result.metrics['mean_coverage_rate'])
                env.close()
            coverages[spec] = float(np.mean(scores))
        self.assertGreater(
            coverages['oracle'],
            coverages['random'],
            f'oracle planner did not beat random: {coverages}',
        )


@unittest.skipUnless(SLOW, 'set RESEARCH_SLOW_TESTS=1 to run the full pipeline test')
class TestEndToEnd(unittest.TestCase):
    """Brief section 32/33: the whole pipeline, including DIMA's modules."""

    def test_smoke_test_passes(self):
        from research.smoke_test import main

        self.assertEqual(main(['--max-episode-steps', '20']), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
