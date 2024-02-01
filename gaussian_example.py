import os
import math
import random
import time
import pickle

import numpy as np
import cv2
import imageio

import argparse
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0., num_cycles: float = 0.5):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def sample_gassian(mu, sigma, N_samples=None, seed=None):
    assert N_samples is not None or seed is not None
    if seed is None:
        seed = torch.randn((N_samples, d), device=mu.device)
    samples = mu + torch.matmul(seed, sigma.t())
    return samples

# Core function: compute score function of perturbed Gaussian distribution
# \nabla \log p_t(x_t) = -(Simga^{-1} + sigma_t^2 I) (x_t - \alpha_t * \mu)
def calc_perturbed_gaussian_score(x, mu, sigma, alpha_noise, sigma_noise):
    if mu.ndim == 1:
        mu = mu[None, ...] # [d] -> [1, d]
    if sigma.ndim == 2:
        sigma = sigma[None, ...] # [d, d] -> [1, d, d]

    mu = mu * alpha_noise[..., None] # [B, d]
    sigma = torch.matmul(sigma, sigma.permute(0, 2, 1)) # [1, d, d]
    sigma = (alpha_noise**2)[..., None, None] * sigma # [B, d, d]
    sigma = sigma + (sigma_noise**2)[..., None, None] * torch.eye(sigma.shape[1], device=sigma.device)[None, ...] # [B, d, d]
    inv_sigma = torch.inverse(sigma) # [B, d, d]
    return torch.matmul(inv_sigma, (mu - x)[..., None]).squeeze(-1) # [B, d, d] @ [B, d, 1] -> [B, d, 1] -> [B, d]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--method', required=True, type=str, choices=['sds', 'vsd', 'esd'], help='score distillation method')
    parser.add_argument('--output_dir', default='./output', type=str, help='directory path to save outputs')
    parser.add_argument('--ndim', default=2, type=int, help='dimension of Gaussian examples')
    parser.add_argument('--dist_0', default=10., type=float, help='distance between initialization to the GT')
    parser.add_argument('--lambda_coeff', default=1.0, type=float, help='coefficient lambda for ESD.')
    parser.add_argument('--seed', default=0, type=int, help='global seed')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of total optimization steps')
    parser.add_argument('--warmup_steps', default=100, type=int, help='number of warm-up optimization steps')
    parser.add_argument('--batch_size', default=256, type=int, help='optimization batch size')
    parser.add_argument('--lr', default=1e-2, type=float, help='optimization learning rate')
    parser.add_argument('--min_lr', default=0., type=float, help='optimization learning rate')
    parser.add_argument('--weight_decay', default=0., type=float, help='optimization weight decay')
    parser.add_argument('--scheduler_type', default='cosine', type=str, help='lr scheduler type')
    parser.add_argument('--logging_steps', default=10, type=int, help='logging for every x steps')
    parser.add_argument('--num_samples_vis', default=1000, type=int, help='number of total optimization steps')
    parser.add_argument('--device', default='0', type=str, help='GPU/CPU device to run experiment')
    parser.add_argument('--save_video', default=True, action='store_true', help='visualize training process as video/gif')
    parser.add_argument('--fps_video', default=30, type=int, help='FPS to export video/gif')
    args = parser.parse_args()

    # data dimension
    N = args.batch_size
    d = args.ndim

    # create directory for output
    os.makedirs(args.output_dir, exist_ok=True)

    # setup running device, random seeds, and plot parameters
    device = torch.device(f'cuda:{int(args.device)}') if args.device.isdigit() else str(args.device)
    seed_everything(args.seed)

    # groundtruth distribution
    p_mu = torch.rand(d, device=device) # uniform random in [0, 1] x [0, 1]
    p_sigma = torch.rand((d, d), device=device) + torch.eye(d, device=device) # positive semi-definite

    # diffusion coefficients
    beta_start = 0.0001
    beta_end = 0.02

    # parametric distribution to optimize
    q_mu = nn.Parameter(torch.rand(d, device=device) * args.dist_0 + p_mu)
    q_sigma = nn.Parameter(torch.rand(d, d, device=device))

    optimizer = torch.optim.AdamW([q_mu, q_sigma], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, int(args.total_steps*1.5), args.min_lr) if args.scheduler_type == 'cosine' else None

    # saving checkpoints
    state_dict = []

    # store per-step samples. fixed seed for visualization
    vis_seed = torch.randn((2, N, d), device=device)
    vis_samples = [] # [steps, p+q, N_samples, N_dim]

    for i in trange(args.total_steps + 1):
        optimizer.zero_grad()

        # sample time steps and compute noise coefficients
        betas_noise = torch.rand(N, device=device) * (beta_end - beta_start) + beta_start
        alphas_noise = torch.cumprod(1.0 - betas_noise, dim=0)
        sigmas_noise = ((1 - alphas_noise) / alphas_noise) ** 0.5

        # sample from g(x) = q_mu + q_sigma @ c, c ~ N(0, I)
        x = sample_gassian(q_mu, q_sigma, N_samples=N)
        # sample gaussian noise
        eps = torch.randn((N, d), device=device)
        # diffuse and perturb samples
        x_t = x * alphas_noise[..., None] + eps * sigmas_noise[..., None]

        # w(t) coefficients
        w = ((1 - alphas_noise) * sigmas_noise)[..., None]

        # compute score distillation update
        with torch.no_grad():
            # \nabla \log p_t(x_t)
            score_p = calc_perturbed_gaussian_score(x_t, p_mu, p_sigma, alphas_noise, sigmas_noise)

            if args.method == 'sds':
                # -[\nabla \log p_t(x_t) - eps]
                grad = -w * (score_p - eps)
            elif args.method == 'vsd':
                # \nabla \log q_t(x_t | c) - centering trick
                cond_mu = x.detach()
                cond_sigma = torch.zeros_like(q_sigma)
                score_q = calc_perturbed_gaussian_score(x_t, cond_mu, cond_sigma, alphas_noise, sigmas_noise)

                # -[\nabla \log p_t(x_t) - \nabla \log q_t(x_t | c)]
                grad = -w * (score_p - score_q)
            elif args.method == 'esd':
                # \nabla \log q_t(x_t)
                score_q_uncond = calc_perturbed_gaussian_score(x_t, q_mu, q_sigma, alphas_noise, sigmas_noise)

                # \nabla \log q_t(x_t | eps) - centering trick
                cond_mu = x.detach()
                cond_sigma = torch.zeros_like(q_sigma)
                score_q_cond = calc_perturbed_gaussian_score(x_t, cond_mu, cond_sigma, alphas_noise, sigmas_noise)

                # -[\nabla \log p_t(x_t) - \lambda \nabla \log q_t(x_t) - (1 - \lambda) \nabla \log q_t(x_t | eps)]
                grad = -w * (score_p - args.lambda_coeff * score_q_uncond - (1. - args.lambda_coeff) * score_q_cond)
    

        # reparameterization trick for backpropagation
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        grad = torch.nan_to_num(grad)
        target = (x_t - grad).detach()
        loss = 0.5 * F.mse_loss(x_t, target, reduction="sum") / N

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # logging
        if i % args.logging_steps == 0:
            state_dict.append({
                'step': i,
                'q_mu': q_mu.detach().cpu().numpy(),
                'q_sigma': q_sigma.detach().cpu().numpy(),
            })

            # save sample positions
            with torch.no_grad():
                p_samples = sample_gassian(p_mu, p_sigma, seed=vis_seed[0])
                p_samples = p_samples.detach().cpu().numpy()
                
                q_samples = sample_gassian(q_mu, q_sigma, seed=vis_seed[1])
                q_samples = q_samples.detach().cpu().numpy()
    
                vis_samples.append(np.stack([p_samples, q_samples], 0))


    # save checkpoints
    with open(os.path.join(args.output_dir, 'state_dict.pkl'), 'wb') as f:
        pickle.dump(state_dict, f)

    # prepare data and arguments for plotting
    if args.ndim > 2:
        print('WARNING: Currently only support plotting 2D figures. Only first two dimension is plotted.')

    vis_samples = np.stack(vis_samples, 0)
    title = args.method.upper() if args.method in ['sds', 'vsd'] else f'{args.method.upper()}($\\lambda={args.lambda_coeff}$)'
    p_plt_kwargs = dict(c='tab:blue', edgecolors='snow', linewidth=0.1, marker='o', alpha=1.)
    q_plt_kwargs = dict(c='tab:orange', edgecolors='snow', linewidth=0.1, marker='X', alpha=1.)

    # plot the final result
    fig, ax = plt.subplots(1, figsize=(5, 5))
    p_ax = ax.scatter(vis_samples[-1, 0, :, 0], vis_samples[-1, 0, :, 1], **p_plt_kwargs)
    q_ax = ax.scatter(vis_samples[-1, 1, :, 0], vis_samples[-1, 1, :, 1], **q_plt_kwargs)
    ax.grid(True)
    ax.legend((p_ax, q_ax), ('GT Dist. $p(x, y)$', 'Fitted Dist. $q(x, y)$'), ncol=1, loc='best', scatterpoints=1)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.savefig(os.path.join(args.output_dir, f'final_result.png'))
    plt.close(fig)

    if args.save_video:

        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
        fig, ax = plt.subplots(1, figsize=(5, 5))

        # adjust axis range
        xlim = (np.min(vis_samples[..., 0]), np.max(vis_samples[..., 0]))
        ylim = (np.min(vis_samples[..., 1]), np.max(vis_samples[..., 1]))
        w, h = xlim[1] - xlim[0], ylim[1] - ylim[0]
        xlim = (xlim[0]-w/4., xlim[1]+w/4.)
        ylim = (ylim[0]-w/4., ylim[1]+w/4.)

        pbar = tqdm(total=vis_samples.shape[0], desc='Converting videos')
        
        def animate_func(i):
            ax.clear()
            p_ax = ax.scatter(vis_samples[i, 0, :, 0], vis_samples[i, 0, :, 1], **p_plt_kwargs)
            q_ax = ax.scatter(vis_samples[i, 1, :, 0], vis_samples[i, 1, :, 1], **q_plt_kwargs)
            ax.set_title(title + f" - Step {state_dict[i]['step']}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend((p_ax, q_ax), ('GT Dist. $p(x, y)$', 'Fitted Dist. $q(x, y)$'), ncol=1, loc='lower right', scatterpoints=1)
            pbar.update()

        ani = FuncAnimation(fig, animate_func, frames=vis_samples.shape[0], interval=10000, repeat=False)
        ani.save(os.path.join(args.output_dir, "trajectory.gif"), dpi=300, writer=PillowWriter(fps=args.fps_video))
        
        plt.close(fig)

