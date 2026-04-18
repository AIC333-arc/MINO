import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('./')
sys.path.append('./models')

from util.util import make_2d_grid
from util.conditional_data import PerThetaSPDEDataset, nested_collate
from util.ofm_OT_likelihood_seq_mino import OFMModel

from models.mino_transformer import MINO
from models.mino_modules.decoder_perceiver import DecoderPerceiver
from models.mino_modules.encoder_supernodes_gno_cross_attention import EncoderSupernodes
from models.mino_modules.conditioner_timestep_theta import ConditionerTimestepTheta


parser = argparse.ArgumentParser('Conditional MINO-T on Heston SPDE')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--spath', type=str, required=True)

# Domain
parser.add_argument('--x_dim', type=int, default=2)             # (K, tau)
parser.add_argument('--query_dims', type=int, nargs='+', default=[16, 16])
parser.add_argument('--co_domain', type=int, default=1)
parser.add_argument('--radius', type=float, default=0.1)
parser.add_argument('--theta_dim', type=int, default=5)          # Heston

# GP prior
parser.add_argument('--kernel_length', type=float, default=0.01)
parser.add_argument('--kernel_variance', type=float, default=1.0)
parser.add_argument('--nu', type=float, default=0.5)
parser.add_argument('--sigma_min', type=float, default=1e-4)

# Paper hyperparameters (Appendix B.1, B.2)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--enc_depth', type=int, default=4)
parser.add_argument('--dec_depth', type=int, default=4)
parser.add_argument('--B_theta', type=int, default=32)           # theta groups per step
parser.add_argument('--M_pairs', type=int, default=32)           # pairs per theta
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--step_size', type=int, default=25)
parser.add_argument('--gamma', type=float, default=0.8)

# Evaluation
parser.add_argument('--n_eval_samples', type=int, default=500,
                    help='sampled functions per test theta')
parser.add_argument('--eval', type=int, default=0)

args = parser.parse_args()
spath = Path(args.spath)
spath.mkdir(parents=True, exist_ok=True)


def main():
    # Data
    query_pos = make_2d_grid(args.query_dims)                    # (d, n_query)
    query_pos = query_pos.permute(1, 0)                          # (n_query, d)

    train_ds = PerThetaSPDEDataset(args.data_path, 'train', query_pos=query_pos)
    loader_tr = DataLoader(
        dataset=train_ds,
        batch_size=args.B_theta,
        shuffle=True,
        collate_fn=nested_collate(M=args.M_pairs),
        num_workers=2,
    )
    print(f'Loaded train: K={len(train_ds)}, '
          f'batch = {args.B_theta} theta x {args.M_pairs} pairs per step')

    coords_np = np.load(os.path.join(args.data_path, 'coords.npy')).astype(np.float32)
    n_pos = torch.from_numpy(coords_np)                          # (n_points, d)

    # Model
    conditioner = ConditionerTimestepTheta(dim=args.dim, theta_dim=args.theta_dim)

    model = MINO(
        conditioner=conditioner,
        encoder=EncoderSupernodes(
            input_dim=args.co_domain,
            ndim=args.x_dim,
            radius=args.radius,
            enc_dim=args.dim,
            enc_num_heads=args.num_heads,
            enc_depth=args.enc_depth,
            cond_dim=conditioner.cond_dim,
        ),
        decoder=DecoderPerceiver(
            input_dim=args.dim,
            output_dim=args.co_domain,
            ndim=args.x_dim,
            dim=args.dim,
            num_heads=args.num_heads,
            depth=args.dec_depth,
            unbatch_mode='dense_to_sparse_unpadded',
            cond_dim=conditioner.cond_dim,
        ),
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'MINO parameters: {n_params:.1f}M')

    # Train/Load
    fmot = OFMModel(
        model,
        kernel_length=args.kernel_length,
        kernel_variance=args.kernel_variance,
        nu=args.nu,
        sigma_min=args.sigma_min,
        device=args.device,
        x_dim=args.x_dim,
        n_pos=n_pos,
    )

    if args.eval:
        for p in model.parameters():
            p.requires_grad = False
        ckpt = torch.load(spath / f'epoch_{args.epochs}.pt',
                          map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt, strict=False)
    else:
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
        fmot.train(
            loader_tr, optimizer,
            epochs=args.epochs, scheduler=scheduler,
            eval_int=0, save_int=args.epochs, generate=False,
            save_path=spath, saved_model=1,
        )
        print('Training complete.')

    # Sampling over test thetas
    for split in ['test_in_dist', 'test_crisis']:
        theta_test = torch.from_numpy(
            np.load(os.path.join(args.data_path, f'theta_{split}.npy')).astype(np.float32)
        )
        all_samples = sample_over_thetas(fmot, theta_test, n_pos, query_pos, args)
        np.save(spath / f'samples_{split}.npy', all_samples.numpy())
        np.save(spath / f'theta_{split}.npy', theta_test.numpy())
        print(f'{split}: sampled {all_samples.shape}')


def sample_over_thetas(fmot, theta_test, n_pos, query_pos, args):
    """For each test theta, draw n_eval_samples from the learned conditional law."""
    all_samples = []
    n_eval = args.n_eval_samples
    with torch.no_grad():
        for i in range(len(theta_test)):
            theta_star = theta_test[i:i+1].repeat(n_eval, 1).to(args.device)
            pos = n_pos.unsqueeze(0).repeat(n_eval, 1, 1).permute(0, 2, 1).to(args.device)
            qpos = query_pos.unsqueeze(0).repeat(n_eval, 1, 1).permute(0, 2, 1).to(args.device)

            samples = fmot.sample(
                pos=pos, query_pos=qpos, theta=theta_star,
                n_samples=n_eval, n_channels=args.co_domain, n_eval=2,
            ).cpu()
            all_samples.append(samples)

    # (K_test, n_eval_samples, C, n_points)
    return torch.stack(all_samples, dim=0)


if __name__ == '__main__':
    main()