from torch.utils.data import Sampler
import numpy as np

class StratifiedSampler(Sampler):
    """Stratified Sampling without replacement."""
    def __init__(self, labels):
        self.num_samples = len(labels)
        self.class_vector = np.array(labels).astype(int)
        self.indices = np.arange(self.num_samples)
        
    def __iter__(self):
        # Find the unique classes and their respective counts
        class_counts = np.bincount(self.class_vector)
        class_indices = [np.where(self.class_vector == i)[0] for i in range(len(class_counts))]
        
        # Calculate the number of items per class in each batch
        n_batches = self.num_samples // len(class_counts)
        indices = []
        for _ in range(n_batches):
            for class_idx in class_indices:
                indices.append(class_idx[np.random.randint(len(class_idx))])
        np.random.shuffle(indices)
        
        # In case the total number of samples isn't divisible by the number of classes,
        # randomly select the remaining indices
        remainder = self.num_samples - len(indices)
        if remainder:
            extra_indices = np.random.choice(self.indices, size=remainder, replace=False)
            indices = np.concatenate([indices, extra_indices])
        
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples