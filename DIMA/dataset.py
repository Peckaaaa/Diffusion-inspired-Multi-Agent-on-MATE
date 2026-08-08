from collections import deque
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import pickle
import psutil
import torch
import torch.nn.functional as F
import numpy as np
import torch.utils
import torch.utils.data

from episode import Episode, SC2Episode

import ipdb

@dataclass
class Batch:
    obs: torch.FloatTensor
    shared_obs: torch.FloatTensor
    next_shared_obs: torch.FloatTensor
    act: torch.LongTensor
    av_action: torch.FloatTensor
    rew: torch.FloatTensor
    end: torch.FloatTensor
    mask_padding: torch.BoolTensor

    def to(self, device: torch.device):
        return Batch(**{k: v.to(device) if v is not None else v for k, v in self.__dict__.items()})

### TODO:
def convert_to_batch(marie_batch):
    import ipdb; ipdb.set_trace()

    attrs = ("observation", "shared_obs", "action", "av_action", "reward", "done", "filled")
    stack = (torch.stack([getattr(s, x) for s in episodes_segments]) for x in attrs)
    return Batch(*stack)


class EpisodesDataset:
    def __init__(self, max_num_episodes: Optional[int] = None, name: Optional[str] = None) -> None:
        self.max_num_episodes = max_num_episodes
        self.name = name if name is not None else 'dataset'
        self.num_seen_episodes = 0
        self.episodes = deque()
        self.visit_entries = deque()
        self.min_episode_length = 20

        self.sample_temperature = 20.

        self.episode_id_to_queue_idx = dict()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def __len__(self) -> int:
        return len(self.episodes)

    def clear(self) -> None:
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()

    def add_episode(self, episode: Union[Episode, SC2Episode]) -> int:
        if self.max_num_episodes is not None and len(self.episodes) == self.max_num_episodes:
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def get_episode(self, episode_id: int) -> Episode:
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        return self.episodes[queue_idx]

    def update_episode(self, episode_id: int, new_episode: Episode) -> None:
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        merged_episode = self.episodes[queue_idx].merge(new_episode) # merge concatenates old and new episodes along dim 0
        self.episodes[queue_idx] = merged_episode
        self.newly_modified_episodes.add(episode_id)

    # Pop episode from the front of the queue
    def _popleft(self) -> Episode:
        id_to_delete = [k for k, v in self.episode_id_to_queue_idx.items() if v == 0]
        assert len(id_to_delete) == 1
        self.newly_deleted_episodes.add(id_to_delete[0])
        self.episode_id_to_queue_idx = {k: v - 1 for k, v in self.episode_id_to_queue_idx.items() if v > 0}
        self.visit_entries.popleft()
        return self.episodes.popleft()

    def _append_new_episode(self, episode):
        episode_id = self.num_seen_episodes
        self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
        self.episodes.append(episode) # Store episode at the end of the queue
        self.visit_entries.append(0.)
        if len(episode) < self.min_episode_length:
            self.min_episode_length = len(episode)

        self.num_seen_episodes += 1
        self.newly_modified_episodes.add(episode_id)
        return episode_id

    def sample_batch(self, batch_num_samples: int, sequence_length: int, sample_from_start: bool = True, valid_sample: bool = False) -> Batch:
        return self._collate_episodes_segments(self._sample_episodes_segments(batch_num_samples, sequence_length, sample_from_start, valid_sample))

    def _sample_episodes_segments(self, batch_num_samples: int, sequence_length: int, sample_from_start: bool, valid_sample: bool) -> List[Episode]:
        sampled_episodes = random.choices(self.episodes, k=batch_num_samples)
        sampled_episodes_segments = []
        for sampled_episode in sampled_episodes:
            if not valid_sample:
                if sample_from_start:
                    start = random.randint(0, len(sampled_episode) - 1)
                    stop = start + sequence_length
                else:
                    stop = random.randint(1, len(sampled_episode))
                    start = stop - sequence_length
            else:
                start = random.randint(0, len(sampled_episode) - sequence_length)
                stop = start + sequence_length

            sampled_episodes_segments.append(sampled_episode.segment(start, stop, should_pad=True))
            assert len(sampled_episodes_segments[-1]) == sequence_length
        return sampled_episodes_segments
        # sampled_episodes_segments = []
        # sample_probs = np.exp(- np.array(self.visit_entries) / self.sample_temperature) / np.exp(- np.array(self.visit_entries) / self.sample_temperature).sum()

        # for i in range(batch_num_samples):
        #     while True:
        #         rand_idx = int(np.random.choice(len(self.episodes), 1, p=sample_probs))
        #         sampled_episode = self.episodes[rand_idx]
        #         if valid_sample:
        #             if len(sampled_episode) - sequence_length > 0:
        #                 break
        #         else:
        #             break
            
        #     # self.visit_entries[rand_idx] += 1
        #     if not valid_sample:
        #         if sample_from_start:
        #             start = random.randint(0, len(sampled_episode) - 1)
        #             stop = start + sequence_length
        #         else:
        #             stop = random.randint(1, len(sampled_episode))
        #             start = stop - sequence_length
        #     else:
        #         start = random.randint(0, len(sampled_episode) - sequence_length)
        #         stop = start + sequence_length

        #     sampled_episodes_segments.append(sampled_episode.segment(start, stop, should_pad=True))
        #     assert len(sampled_episodes_segments[-1]) == sequence_length
        
        # return sampled_episodes_segments

    def _collate_episodes_segments(self, episodes_segments: List[Episode]) -> Batch:
        episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
        
        stack = (torch.stack([e_s[k] for e_s in episodes_segments]) for k in episodes_segments[0])
        import ipdb; ipdb.set_trace()

        return Batch(*stack)
        batch = {}
        for k in episodes_segments[0]:
            batch[k] = torch.stack([e_s[k] for e_s in episodes_segments]) # 
        batch['observations'] = batch['observations'].float() / 255.0  # int8 to float and scale to [0, 1]
        return batch

    def traverse(self, batch_num_samples: int, chunk_size: int):
        for episode in self.episodes:
            chunks = [episode.segment(start=i * chunk_size, stop=(i + 1) * chunk_size, should_pad=True) for i in range(math.ceil(len(episode) / chunk_size))]
            batches = [chunks[i * batch_num_samples: (i + 1) * batch_num_samples] for i in range(math.ceil(len(chunks) / batch_num_samples))]
            for b in batches:
                yield self._collate_episodes_segments(b)

    def update_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir()
        for episode_id in self.newly_modified_episodes:
            episode = self.get_episode(episode_id)
            episode.save(directory / f'{episode_id}.pt')
        for episode_id in self.newly_deleted_episodes:
            (directory / f'{episode_id}.pt').unlink()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def load_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir() and len(self.episodes) == 0
        episode_ids = sorted([int(p.stem) for p in directory.iterdir()])
        self.num_seen_episodes = episode_ids[-1] + 1
        for episode_id in episode_ids:
            episode = Episode(**torch.load(directory / f'{episode_id}.pt'))
            self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
            self.episodes.append(episode)


class EpisodesDatasetRamMonitoring(EpisodesDataset):
    """
    Prevent episode dataset from going out of RAM.
    Warning: % looks at system wide RAM usage while G looks only at process RAM usage.
    """
    def __init__(self, max_ram_usage: str, name: Optional[str] = None) -> None:
        super().__init__(max_num_episodes=None, name=name)
        self.max_ram_usage = max_ram_usage
        self.num_steps = 0
        self.max_num_steps = None

        self.episode_lens = deque()

        max_ram_usage = str(max_ram_usage)
        if max_ram_usage.endswith('%'):
            m = int(max_ram_usage.split('%')[0])
            assert 0 < m < 100
            self.check_ram_usage = lambda: psutil.virtual_memory().percent > m
        else:
            assert max_ram_usage.endswith('G')
            m = float(max_ram_usage.split('G')[0])
            self.check_ram_usage = lambda: psutil.Process().memory_info()[0] / 2 ** 30 > m

    def clear(self) -> None:
        super().clear()
        self.num_steps = 0

    def add_episode(self, episode: Episode) -> int:
        if self.max_num_steps is None and self.check_ram_usage():
            self.max_num_steps = self.num_steps
        self.num_steps += len(episode)
        self.episode_lens.append(len(episode))
        while (self.max_num_steps is not None) and (self.num_steps > self.max_num_steps):
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def _popleft(self) -> Episode:
        episode = super()._popleft()
        self.episode_lens.popleft()
        self.num_steps -= len(episode)
        return episode


class MultiAgentEpisodesDataset(EpisodesDatasetRamMonitoring):
    def __init__(self, max_ram_usage: str, name: Optional[str] = None, sample_weights: Optional[List[float]] = None,
                 capacity: int = None, diffusion_seq_len: int = None, condition_steps: int = None, sample_temp = 'inf') -> None:
        super().__init__(max_ram_usage, name)
        # self.sample_visits = torch.zeros(100000, dtype=torch.long, device='cpu')
        self.sample_weights = sample_weights

        ## only serve for sampling segment for diffusion model learning
        self.diffusion_seq_len = diffusion_seq_len
        self.condition_steps = condition_steps
        self.sample_temp = sample_temp
        self.capacity = capacity

        self.segment_ids  = torch.empty((self.capacity, 3), dtype=torch.long)
        self.visit_counts = torch.zeros(self.capacity, dtype=torch.long)
        self.next_idx = 0
        self.full = False

    def add_episode(self, episode):
        if self.max_num_steps is None and self.check_ram_usage():
            self.max_num_steps = self.num_steps
        self.num_steps += len(episode)
        self.episode_lens.append(len(episode))
        while (self.max_num_steps is not None) and (self.num_steps > self.max_num_steps):
            self._popleft()
        episode_id = self._append_new_episode(episode)

        num_segments = len(episode) - self.diffusion_seq_len + self.condition_steps
        epi_ids = torch.ones(num_segments, dtype=torch.long) * episode_id
        starts = torch.arange(num_segments, dtype=torch.long) - self.condition_steps + 1
        ends = starts + self.diffusion_seq_len

        cur_segment_ids = torch.stack([epi_ids, starts, ends], dim=1)
        
        if self.next_idx + num_segments > self.capacity:
            self.segment_ids[self.next_idx : self.capacity] = cur_segment_ids[: self.capacity - self.next_idx]
            self.visit_counts[self.next_idx : self.capacity] = self.visit_counts[self.next_idx : self.capacity] * 0

            self.next_idx = (self.next_idx + num_segments) % self.capacity
            self.segment_ids[:self.next_idx] = cur_segment_ids[-self.next_idx :]
            self.visit_counts[:self.next_idx] = self.visit_counts[:self.next_idx] * 0

            self.full = True

        else:
            self.segment_ids[self.next_idx : self.next_idx + num_segments] = cur_segment_ids
            self.visit_counts[self.next_idx : self.next_idx + num_segments] = self.visit_counts[self.next_idx : self.next_idx + num_segments] * 0

            self.next_idx = (self.next_idx + num_segments) % self.capacity

        return episode_id
        
    def _compute_visit_probs(self, n):
        if self.sample_temp == 'inf':
            visits = self.visit_counts[:n].float()
            visit_sum = visits.sum()
            if visit_sum == 0:
                probs = torch.full_like(visits, 1 / n)
            else:
                probs = 1 - visits / visit_sum
        else:
            logits = self.visit_counts[:n].float() / -self.sample_temp
            probs = F.softmax(logits, dim=0)
        assert probs.device.type == 'cpu'
        return probs

    # NOTE: only serve for diffusion model learning
    def sample_batch_new(self, batch_num_samples):
        probs = self._compute_visit_probs(self.capacity if self.full else self.next_idx)
        segment_indices = torch.multinomial(probs, batch_num_samples, replacement=False)
        
        # stay on cpu
        flat_idx = segment_indices.reshape(-1)
        flat_idx, counts = torch.unique(flat_idx, return_counts=True)
        self.visit_counts[flat_idx] += counts
        
        segment_indices = segment_indices.cpu()
        sampled_segment_ids = self.segment_ids[segment_indices]
        
        sampled_episodes_segments = []
        for idx in range(len(sampled_segment_ids)):
            epi_id, st, end = sampled_segment_ids[idx]
            sampled_episodes_segments.append(self.episodes[epi_id].segment(st, end, should_pad=True))
        
        return self._collate_episodes_segments(sampled_episodes_segments)

    def _sample_episodes_segments(self, batch_num_samples, sequence_length, can_sample_beyond_end, valid_sample):
        num_episodes = len(self.episodes)
        episode_lens = np.array(list(self.episode_lens))

        sampled_episodes_segments = []

        if (self.sample_weights is None) or num_episodes < len(self.sample_weights):
            weights = episode_lens / self.num_steps
        else:
            weights = self.sample_weights
            num_weights = len(self.sample_weights)
            assert all([0 <= x <= 1 for x in weights]) and sum(weights) == 1
            sizes = [
                num_episodes // num_weights + (num_episodes % num_weights) * (i == num_weights - 1)
                for i in range(num_weights)
            ]
            weights = [w / s for (w, s) in zip(weights, sizes) for _ in range(s)]
        
        episodes_partition = np.arange(0, num_episodes, 1)
        weights = np.array(weights[0::1])
        episode_ids = np.random.choice(episodes_partition, size=batch_num_samples, replace=True, p=weights / weights.sum())   # batch_size: first select episodes, then truncate
        timesteps = np.random.randint(low=0, high=episode_lens[episode_ids])    # Left-closed, right-open, range is [0, chosen_episode_lengths - 1]

        if not valid_sample:
            if can_sample_beyond_end:
                starts = timesteps - np.random.randint(0, sequence_length, len(timesteps))  # Second term range is [0, sequence_length - 1], combined range is [- sequence_length + 1, chosen_episode_lengths - 1]
                stops = starts + sequence_length
            
            else:
                # minimum second term range is [1, chosen_episode_lengths + sequence_length - 1]
                stops = np.minimum(
                    episode_lens[episode_ids], timesteps + 1 + np.random.randint(0, sequence_length, len(timesteps))
                )
                starts = stops - sequence_length

        else:
            starts = np.random.randint(low=0, high=episode_lens[episode_ids] - sequence_length)
            stops = starts + sequence_length
        
        for idx, (st, end) in enumerate(zip(starts, stops)):
            sampled_episodes_segments.append(self.episodes[episode_ids[idx]].segment(st, end, should_pad=True))

        return sampled_episodes_segments

    # def _sample_episodes_segments(self, batch_num_samples: int, sequence_length: int, sample_from_start: bool, valid_sample: bool) -> List[Episode]:
    #     # updated samplling indices
    #     n = len(self.episodes)
    #     probs = self._compute_visit_probs(n)
    #     start_idx = torch.multinomial(probs, batch_num_samples, replacement=True)
        
    #     ipdb.set_trace()
        
    #     # stay on cpu
    #     flat_idx = start_idx.reshape(-1)
    #     flat_idx, counts = torch.unique(flat_idx, return_counts=True)
    #     self.sample_visits[flat_idx] += counts
        
    #     ipdb.set_trace()
        
    #     sampled_episodes_segments = []
    #     for sampled_episode in sampled_episodes:
    #         if not valid_sample:
    #             if sample_from_start:
    #                 start = random.randint(0, len(sampled_episode) - 1)
    #                 stop = start + sequence_length
    #             else:
    #                 stop = random.randint(1, len(sampled_episode))
    #                 start = stop - sequence_length
    #         else:
    #             start = random.randint(0, len(sampled_episode) - sequence_length)
    #             stop = start + sequence_length

    #         sampled_episodes_segments.append(sampled_episode.segment(start, stop, should_pad=True))
    #         assert len(sampled_episodes_segments[-1]) == sequence_length
    #     return sampled_episodes_segments
    

    def _collate_episodes_segments(self, episodes_segments: List[Episode]) -> Batch:
        # episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
        attrs = list(episodes_segments[0].__dict__.keys())
        # attrs = ("observation", "shared_obs", "action", "av_action", "reward", "done", "filled")
        # stack = {x: torch.stack([getattr(s, x) for s in episodes_segments]) for x in attrs}
        stack = {}
        for x in attrs:
            stack[x] = torch.stack([getattr(s, x) for s in episodes_segments])

        return Batch(
            obs = stack.get('observation'),
            shared_obs = stack.get('shared_obs'),
            next_shared_obs = stack.get('next_shared_obs'),
            act = stack.get('action'),
            av_action = stack.get('av_action', None),
            rew = stack.get('reward'),
            end = stack.get('done'),
            mask_padding = stack.get('filled'),
        )
    
    def load_from_pkl(self, dataset_path):
        '''
        pre-loading buffer, but we filter out absorbing state
        '''
        # loading data
        f = open(dataset_path, 'rb+')
        data = pickle.load(f)
        f.close()
        
        # preprocess data
        valid_indices = np.argwhere(data["fakes"].all(-2).squeeze() == False).squeeze().tolist()
        observations = data["observations"][valid_indices]
        actions = data["actions"][valid_indices]
        rewards = data["rewards"][valid_indices]
        av_actions = data["av_actions"][valid_indices]
        dones = data["dones"][valid_indices]
        
        num_steps = dones.shape[0]
        
        dones_indices = np.argwhere(dones.all(-2).squeeze() == True).squeeze().tolist()
        start = 0
        for idx in dones_indices:
            episode = SC2Episode(
                observation=torch.FloatTensor(observations[start : idx + 1]),
                action=torch.FloatTensor(actions[start : idx + 1]),
                av_action=torch.FloatTensor(av_actions[start : idx + 1]),
                reward=torch.FloatTensor(rewards[start : idx + 1]),
                done=torch.FloatTensor(dones[start : idx + 1]),
                filled=torch.ones(idx + 1 - start, dtype=torch.bool)
            )

            self.add_episode(episode)
            
            start = idx + 1

        print(f"{self.num_steps} environment steps have been loaded.")
        print(f"{len(self.episodes)} episodes have been loaded.")


@dataclass
class SegmentId:
    episode_id: int
    start: int
    stop: int

def collate_episodes_segments(episodes_segments: List[Episode]) -> Batch:
    # episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
    attrs = ("observation", "shared_obs", "action", "av_action", "reward", "done", "filled")
    stack = (torch.stack([getattr(s, x) for s in episodes_segments]) for x in attrs)
    return Batch(*stack)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, name, episodes):
        super().__init__()
        self.name = name
        self.episodes = episodes
        self.num_episodes = 0
        self.min_episode_length = np.inf

    def __len__(self,):
        return len(self.episodes)
    
    def __getitem__(self, segment_id: SegmentId):
        return self.episodes[segment_id.episode_id].segment(segment_id.start, segment_id.stop, should_pad = True)

    def add_episode(self, episode: Episode):
        self.episodes.append(episode) # Store episode at the end of the queue
        if len(episode) < self.min_episode_length:
            self.min_episode_length = len(episode)

        self.num_episodes += 1

    def report_ram_usage(self,):
        print(f"Occupied {psutil.Process().memory_info()[0] / 2 ** 30} G.")

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self,
                 dataset,
                 segment_len: int,
                 batch_size: int,
                 padding_len: int,
                 can_sample_beyong_end: bool = False):
        super().__init__(dataset)
        self.dataset = dataset
        self.segment_len = segment_len
        self.batch_size = batch_size
        self.padding_len = padding_len
        self.can_sample_beyong_end = can_sample_beyong_end

        self._generate_segment_ids()

    def __len__(self):
        return len(self.indices)
    
    def __iter__(self):
        import random
        random.shuffle(self.indices)

        return iter(self.indices)

    def _generate_segment_ids(self,):
        self.indices = []

        num_episodes = len(self.dataset)
        for idx, episode in enumerate(self.dataset.episodes):
            episode_len = len(episode)
            if episode_len >= self.segment_len:
                if not self.can_sample_beyong_end:
                    num_segments = episode_len - self.segment_len + 1 + self.padding_len
                else:
                    num_segments = episode_len + self.segment_len - 1
                
                for i in range(num_segments):
                    if not self.can_sample_beyong_end:
                        start_idx = i - self.padding_len
                        end_idx = start_idx + self.segment_len
                        self.indices.append(
                            SegmentId(idx, start_idx, end_idx)
                        )
                    else:
                        start_idx = i - self.segment_len + 1
                        end_idx = start_idx + self.segment_len
                        self.indices.append(
                            SegmentId(idx, start_idx, end_idx)
                        )
        