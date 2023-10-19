import numpy as np
import torch


class SinusoidNShot:
    def __init__(self, batchsz, k_shot, k_query, device=None, cache_size=1000):
        self.batchsz = batchsz
        self.k_shot = k_shot
        self.k_query = k_query
        self.device = device
        self.cache_size = cache_size
        self.indexes = {"train": 0, "test": 0}
        self.dataset_sizes = {"train": 1500, "test": 500}  # indicates number of iterations per epoch, but in fact we have infinite data
        self.datasets_cache = {"train": self.generate_data_cache(), "test": self.generate_data_cache()}

    @staticmethod
    def random_sinusoid(n_trajectories):
        amplitude = np.random.uniform(low=.1, high=5., size=(n_trajectories, 1))
        phase = np.random.uniform(low=0., high=np.pi, size=(n_trajectories, 1))
        return amplitude, phase

    @staticmethod
    def sample_sinusoid(amplitude, phase, n_samples):
        assert len(amplitude) == len(phase), "Amplitude and phase must have the same length"
        x = np.random.uniform(low=-5., high=5., size=(len(amplitude), n_samples))
        y = amplitude * np.sin(x + phase)
        return x, y
    
    def generate_batch(self):
        amplitude, phase = self.random_sinusoid(self.batchsz)
        x_spt, y_spt = self.sample_sinusoid(amplitude, phase, self.k_shot)
        x_qry, y_qry = self.sample_sinusoid(amplitude, phase, self.k_query)
        x_spt, y_spt, x_qry, y_qry = [
                torch.from_numpy(z.astype(np.float32)).unsqueeze(-1).to(self.device) for z in
                [x_spt, y_spt, x_qry, y_qry]
        ]
        return x_spt, y_spt, x_qry, y_qry

    def generate_data_cache(self):
        data_cache = []
        for _ in range(self.cache_size):
            data_cache.append(self.generate_batch())    
        return data_cache

    def next(self, mode='train'):
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.generate_data_cache()

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch