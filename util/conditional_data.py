import einops
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class PerThetaSPDEDataset(Dataset):
    """Dataset indexed by theta. Each __getitem__ returns all N_MC realizations
    at a single theta.
    
    Loading: paired .npy files.

    Return shape per item (pre-collation):
        theta:     (p,)
        samples:   (N_MC, C, n_points)  (will be permuted in collator)
        pos:       (n_points, d_spatial)
        query_pos: (n_points_query, d_spatial)
    """

    def __init__(self, data_dir: str, split: str, query_pos: torch.Tensor):
        data_dir = Path(data_dir)
        self.theta = torch.from_numpy(
            np.load(data_dir / f'theta_{split}.npy').astype(np.float32)
        )
        self.samples = torch.from_numpy(
            np.load(data_dir / f'samples_{split}.npy').astype(np.float32)
        )
        coords = np.load(data_dir / 'coords.npy').astype(np.float32)
        self.pos = torch.from_numpy(coords)          # (n_points, d_spatial)
        self.query_pos = query_pos                    # (n_query, d_spatial)

        assert len(self.theta) == len(self.samples), \
            "theta and samples row count must match"

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return dict(
            theta=self.theta[idx],
            samples=self.samples[idx],          # (N_MC, C, n_points)
            pos=self.pos,
            query_pos=self.query_pos,
        )


def nested_collate(M):
    """Returns a collate_fn that draws M samples per theta from each dataset
    item, producing a batch of B theta-groups, M pairs per group.

    Output tensors are flattened to (B*M, ...) for easy PyTorch processing,
    with an added 'group_ids' tensor of shape (B*M,) assigning each row to its
    theta group. The training loop iterates over groups to apply per-theta OT
    coupling.
    """
    def _collate(batch):
        B = len(batch)
        theta_list, data_list, pos_list, qpos_list, group_ids = [], [], [], [], []

        for b, item in enumerate(batch):
            samples = item['samples']              # (N_MC, C, n_points)
            N_MC = samples.shape[0]
            idx = torch.randint(0, N_MC, (M,))     # draw M samples at this theta
            chosen = samples[idx]                  # (M, C, n_points)

            for _ in range(M):
                theta_list.append(item['theta'])
            data_list.append(chosen)
            for _ in range(M):
                pos_list.append(item['pos'])
                qpos_list.append(item['query_pos'])
                group_ids.append(b)

        # Stack to (B*M, ...)
        theta = torch.stack(theta_list, dim=0)                # (B*M, p)
        data = torch.cat(data_list, dim=0)                    # (B*M, C, n_points)
        pos = torch.stack(pos_list, dim=0)                    # (B*M, n_points, d)
        qpos = torch.stack(qpos_list, dim=0)                  # (B*M, n_query, d)
        group_ids = torch.tensor(group_ids, dtype=torch.long) # (B*M,)

        # Permute to MINO's expected layout: (batch, dim, seq_len)
        x_dim = pos.shape[-1]
        C = data.shape[1]
        return dict(
            theta=theta,
            input_feat=einops.rearrange(
                data, 'b c n -> b c n'),                       # already (B*M, C, n_points)
            input_pos=einops.rearrange(
                pos, 'b n d -> b d n'),
            query_pos=einops.rearrange(
                qpos, 'b n d -> b d n'),
            group_ids=group_ids,
        )

    return _collate