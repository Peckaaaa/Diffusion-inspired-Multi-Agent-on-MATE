"""Layered smoke test (brief section 32).

Runs the whole pipeline once and reports which *layer* broke::

    MATE -> adapter -> observation -> DIMA input -> world model
         -> prediction -> planner -> action -> MATE

Every check names the layer it belongs to, so a failure is attributed rather than
just reported.  A missing checkpoint is not a failure: the world-model layers are
then exercised with freshly initialised DIMA modules, which still proves the
plumbing -- shapes, dtypes, devices, decode path -- and is labelled ``untrained``.

    python -m research.smoke_test
    python -m research.smoke_test --checkpoint runs/wm/ckpt/model_final.pth
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.config import configure_torch, default_device
from research.env_adapter import MATEEnv
from research.logging_utils import log
from research.views import ObservationLayout, SceneView


LAYERS = (
    'ENV',
    'ADAPTER',
    'OBSERVATION',
    'DIMA INPUT',
    'WORLD MODEL',
    'PREDICTION',
    'PLANNER',
    'ACTION',
)


@dataclass
class CheckResult:
    layer: str
    name: str
    ok: bool
    detail: str = ''
    error: Optional[str] = None


@dataclass
class SmokeTest:
    scenario: str = 'MATE-4v2-9'
    seed: int = 0
    discrete_levels: int = 5
    max_episode_steps: int = 30
    checkpoint: Optional[str] = None
    device: Optional[str] = None
    results: List[CheckResult] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)

    # -- harness ------------------------------------------------------------ #

    def check(self, layer: str, name: str, fn: Callable[[], str]) -> bool:
        try:
            detail = fn() or ''
        except Exception as exc:  # noqa: BLE001 - the point is to attribute any failure
            self.results.append(
                CheckResult(layer, name, False, error=f'{type(exc).__name__}: {exc}')
            )
            log('FAIL', f'{layer}: {name}')
            traceback.print_exc()
            return False
        self.results.append(CheckResult(layer, name, True, detail=detail))
        log('PASS', f'{layer}: {name}' + (f' -- {detail}' if detail else ''))
        return True

    def run(self) -> bool:
        steps = (
            self._env_reset,
            self._env_step,
            self._observation_shape,
            self._action_shape,
            self._standardized_representation,
            self._dima_input,
            self._world_model,
            self._prediction,
            self._planner,
            self._action_accepted,
        )
        for step in steps:
            if not step():
                return False
        return True

    # -- 1-2. MATE ---------------------------------------------------------- #

    def _env_reset(self) -> bool:
        def run() -> str:
            env = MATEEnv(
                scenario=self.scenario,
                seed=self.seed,
                discrete_levels=self.discrete_levels,
                max_episode_steps=self.max_episode_steps,
            )
            observation = env.reset(seed=self.seed)
            self.state['env'] = env
            self.state['observation'] = observation
            log('ENV', env.describe())
            return f'reset -> obs {observation.obs_raw.shape}, state {observation.state_raw.shape}'

        return self.check('ENV', 'MATE reset', run)

    def _env_step(self) -> bool:
        def run() -> str:
            env = self.state['env']
            action = np.zeros(env.n_agents, dtype=np.int64) + env.n_actions // 2
            observation, reward, done, info = env.step(action)
            self.state['observation'] = observation
            self.state['reward'] = reward
            self.state['info'] = info
            return f'reward {np.round(reward, 4).tolist()} done={done} coverage={info["coverage_rate"]:.3f}'

        return self.check('ENV', 'MATE step', run)

    # -- 3-4. adapter ------------------------------------------------------- #

    def _observation_shape(self) -> bool:
        def run() -> str:
            env, observation = self.state['env'], self.state['observation']
            assert observation.obs_raw.shape == (env.n_agents, env.n_obs), observation.obs_raw.shape
            assert observation.obs.shape == (env.n_agents, env.n_obs)
            assert observation.state_raw.shape == (env.state_dim,)
            assert observation.state.shape == (env.state_dim,)
            assert np.isfinite(observation.obs).all(), 'rescaled observation has non-finite entries'
            roundtrip = np.abs(env.unrescale_obs(observation.obs) - observation.obs_raw).max()
            assert roundtrip < 1e-6, f'rescale round-trip error {roundtrip}'
            return f'obs {observation.obs.shape}, state {observation.state.shape}, rescale round-trip {roundtrip:.2e}'

        return self.check('ADAPTER', 'observation shapes and rescaling', run)

    def _action_shape(self) -> bool:
        def run() -> str:
            env = self.state['env']
            actions = np.arange(env.n_agents) % env.n_actions
            continuous = env.action_to_continuous(actions)
            assert continuous.shape == (env.n_agents, 2), continuous.shape
            recovered = env.action_from_continuous(continuous)
            assert np.array_equal(recovered, actions), (recovered, actions)
            return f'Discrete({env.n_actions}) <-> continuous (rotation, zoom) round-trip exact'

        return self.check('ADAPTER', 'action conversion', run)

    # -- 5. standardized representation ------------------------------------- #

    def _standardized_representation(self) -> bool:
        def run() -> str:
            env, observation = self.state['env'], self.state['observation']
            layout = ObservationLayout.from_env_metadata(env.metadata())
            view = SceneView.from_joint_observation(observation.obs_raw, layout)
            self.state['layout'] = layout
            assert view.camera_positions.shape == (env.n_agents, 2)
            assert view.target_positions.shape == (env.n_targets, 2)

            # The decode is validated against MATE's own coverage_rate, which is
            # computed independently inside the environment.
            estimate = view.coverage_estimate()
            reported = self.state['info']['coverage_rate']
            assert abs(estimate - reported) < 1e-9, (
                f'SceneView coverage {estimate} != MATE coverage {reported}'
            )
            return f'SceneView coverage {estimate:.3f} == MATE coverage_rate {reported:.3f}'

        return self.check('OBSERVATION', 'standardized representation', run)

    # -- 6. DIMA input ------------------------------------------------------ #

    def _dima_input(self) -> bool:
        def run() -> str:
            from research.planners import build_planner
            from research.rollout import run_episode, to_dima_rollout

            env = self.state['env']
            planner = build_planner('reactive_greedy', env, seed=self.seed)
            episode = run_episode(
                env, planner, episode=0, seed=self.seed, max_steps=self.max_episode_steps
            )
            rollout = to_dima_rollout(episode, env.n_actions)
            self.state['episode'] = episode
            self.state['rollout'] = rollout

            steps = len(episode)
            expected = {
                'observation': (steps, env.n_agents, env.n_obs),
                'shared_obs': (steps, env.n_agents, env.state_dim),
                'next_shared_obs': (steps, env.n_agents, env.state_dim),
                'action': (steps, env.n_agents, env.n_actions),
                'reward': (steps, env.n_agents, 1),
                'done': (steps, env.n_agents, 1),
            }
            for key, shape in expected.items():
                assert rollout[key].shape == shape, f'{key}: {rollout[key].shape} != {shape}'
            assert (rollout['action'].sum(-1) == 1).all(), 'actions are not one-hot'
            assert np.isfinite(rollout['shared_obs']).all()

            # The real test: DIMA's own episode/dataset classes must accept it.
            from dataset import MultiAgentEpisodesDataset
            from episode import MamujocoEpisode
            import torch

            dima_episode = MamujocoEpisode(
                observation=torch.FloatTensor(rollout['observation']),
                shared_obs=torch.FloatTensor(rollout['shared_obs']),
                next_shared_obs=torch.FloatTensor(rollout['next_shared_obs']),
                action=torch.FloatTensor(rollout['action']),
                reward=torch.FloatTensor(rollout['reward']),
                done=torch.FloatTensor(rollout['done']),
                filled=torch.ones(steps, dtype=torch.bool),
            )
            dataset = MultiAgentEpisodesDataset(
                max_ram_usage='1G',
                name='smoke',
                capacity=4096,
                diffusion_seq_len=6,
                condition_steps=3,
            )
            dataset.add_episode(dima_episode)
            batch = dataset.sample_batch(
                batch_num_samples=2, sequence_length=6, sample_from_start=False
            )
            assert batch.shared_obs.shape == (2, 6, env.n_agents, env.state_dim), batch.shared_obs.shape
            log('DATA', f'episode length {steps}, DIMA batch {tuple(batch.shared_obs.shape)}')
            return f'{steps} transitions accepted by MamujocoEpisode + MultiAgentEpisodesDataset'

        return self.check('DIMA INPUT', 'DIMA episode and dataset', run)

    # -- 7. world model ----------------------------------------------------- #

    def _world_model(self) -> bool:
        def run() -> str:
            from research.config import build_learner_config
            from research.world_model import DIMAWorldModel

            env = self.state['env']
            config = build_learner_config(env, seed=self.seed, device=self.device, horizon=5)

            if self.checkpoint is None:
                # Build the modules without loading: proves construction, shapes
                # and devices, which is what this layer is responsible for.
                import tempfile
                from pathlib import Path
                import torch
                from agent.learners.DreamerLearner import DreamerLearner

                config.CAPACITY = 4096
                scratch = DreamerLearner(config)
                tmp = Path(tempfile.mkdtemp()) / 'untrained.pth'
                torch.save(scratch.params(), tmp)
                del scratch
                checkpoint = str(tmp)
                label = 'untrained (freshly initialised)'
            else:
                checkpoint = self.checkpoint
                label = self.checkpoint

            world_model = DIMAWorldModel(env, checkpoint, config=config, device=self.device)
            self.state['world_model'] = world_model
            log('WM', f'DIMA world model loaded: {label}')
            log(
                'WM',
                f'conditioning={world_model.conditioning_steps} '
                f'denoising_steps={config.diffusion_sampler_cfg.num_steps_denoising} '
                f'device={world_model.device}',
            )
            return label

        return self.check('WORLD MODEL', 'construct / load DIMA world model', run)

    # -- 8. prediction ------------------------------------------------------ #

    def _prediction(self) -> bool:
        def run() -> str:
            from research.world_model import History

            env = self.state['env']
            world_model = self.state['world_model']
            episode = self.state['episode']

            # History alignment: actions[i] is the action that produced states[i],
            # so the action stored alongside transition k is transitions[k-1].action.
            steps = world_model.conditioning_steps
            history = History(length=steps)
            transitions = episode.transitions
            for offset in range(len(transitions) - steps, len(transitions)):
                history.states.append(transitions[offset].state)
                history.observations.append(transitions[offset].obs)
                history.actions.append(
                    transitions[offset - 1].action if offset > 0 else transitions[offset].action
                )

            candidates = np.tile(episode.transitions[-1].action, (3, 1))
            candidates[:, 0] = [0, env.n_actions // 2, env.n_actions - 1]

            prediction = world_model.predict(history, candidates, horizon=2)
            self.state['prediction'] = prediction

            assert prediction.observations.shape == (3, 2, env.n_agents, env.n_obs), (
                prediction.observations.shape
            )
            assert prediction.states.shape == (3, 2, env.state_dim), prediction.states.shape
            assert np.isfinite(prediction.observations).all(), 'prediction contains NaN/Inf'
            spread = float(
                np.abs(prediction.observations[0] - prediction.observations[-1]).mean()
            )
            log('WM-DIAG', f'observation spread between extreme actions for camera 0: {spread:.4f}')
            return f'obs {prediction.observations.shape}, states {prediction.states.shape}, finite'

        return self.check('PREDICTION', 'world model forward pass', run)

    # -- 9. planner --------------------------------------------------------- #

    def _planner(self) -> bool:
        def run() -> str:
            from research.planners import build_planner
            from research.rollout import _make_context

            env = self.state['env']
            world_model = self.state['world_model']
            episode = self.state['episode']
            layout = self.state['layout']

            planner = build_planner(
                'predictive_greedy', env, world_model=world_model, seed=self.seed, horizon=1
            )
            observation = env.reset(seed=self.seed)

            from research.world_model import History

            history = History(length=world_model.conditioning_steps)
            history.seed(observation, env.n_agents, world_model.noop_action)

            context = _make_context(
                env, planner, layout, observation, history, episode=0, step=0
            )
            planner.reset(observation, context)
            action = planner.plan(observation, None, context)

            assert action.shape == (env.n_agents,), action.shape
            assert ((0 <= action) & (action < env.n_actions)).all(), action
            self.state['planner'] = planner
            self.state['planned_action'] = action
            log('PLANNER', planner.name)
            log('ACTION', ' '.join(f'C{c}=a{a}' for c, a in enumerate(action)))
            log('PLANNER-DIAG', str(planner.diagnostics()))
            return f'action {action.tolist()} within Discrete({env.n_actions})'

        return self.check('PLANNER', 'planner input / output', run)

    # -- 10. back into MATE ------------------------------------------------- #

    def _action_accepted(self) -> bool:
        def run() -> str:
            env = self.state['env']
            observation, reward, done, info = env.step(self.state['planned_action'])
            assert observation.obs_raw.shape == (env.n_agents, env.n_obs)
            return f'MATE accepted the planned action; coverage={info["coverage_rate"]:.3f}'

        return self.check('ACTION', 'MATE accepts the planned action', run)


def summarise(test: SmokeTest) -> None:
    print()
    log('EVAL', 'smoke test summary')
    reached = {result.layer for result in test.results}
    for layer in LAYERS:
        entries = [r for r in test.results if r.layer == layer]
        if not entries:
            status = 'skipped' if layer not in reached else 'unknown'
            print(f'  {layer:<12} {status}')
            continue
        failed = [r for r in entries if not r.ok]
        if failed:
            print(f'  {layer:<12} FAIL   {failed[0].name}: {failed[0].error}')
        else:
            print(f'  {layer:<12} ok     ({len(entries)} check(s))')


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='python -m research.smoke_test', description=__doc__)
    parser.add_argument('--scenario', default='MATE-4v2-9')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discrete-levels', type=int, default=5)
    parser.add_argument('--max-episode-steps', type=int, default=30)
    parser.add_argument('--checkpoint', default=None, help='DIMA world-model checkpoint (.pth)')
    parser.add_argument('--device', default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup = configure_torch(args.device or default_device(), detect_anomaly=False, seed=args.seed)
    log('RUN', f'torch: {setup}')
    test = SmokeTest(
        scenario=args.scenario,
        seed=args.seed,
        discrete_levels=args.discrete_levels,
        max_episode_steps=args.max_episode_steps,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    ok = test.run()
    summarise(test)
    if ok:
        log('PASS', 'MATE -> adapter -> DIMA -> prediction -> planner -> MATE is closed.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
