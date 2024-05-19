import torch
from utils.transforms import transform

class GRUDataset(torch.utils.data.Dataset):
    def __init__(self, structures, targets, max_length, token2id, mean, std):
        super().__init__()
        self.structures = structures
        self.targets = targets
        self.max_length = max_length
        self.token2id = token2id
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, ind):
        struct = self.structures[ind]
        seq_len = len(struct)
        struct = torch.tensor(struct, dtype=torch.int32)
        if struct.shape[0] > self.max_length:
            struct = struct[:max_length]
        elif struct.shape[0] < self.max_length:
            struct = torch.cat([struct, (torch.zeros(self.max_length - struct.shape[0], dtype=torch.int64) + self.token2id['<UNK>'])])
        return struct, torch.tensor(transform(self.targets[ind], self.mean, self.std), dtype=torch.float32), seq_len
    
