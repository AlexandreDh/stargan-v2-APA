"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.utils as vutils

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics

import numpy as np
from pytorch_model_summary import summary


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            print(summary(self.nets.discriminator, torch.zeros([1, 3, args.img_size, args.img_size]), torch.zeros([1], dtype=torch.long)))

            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()

        d_losses_latent_avg = {}
        d_losses_ref_avg = {}
        g_losses_latent_avg = {}
        g_losses_ref_avg = {}

        apa_stat = StatCollector(self.device)

        window_avg_len = 100
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the generator
            g_loss, g_losses_latent, p_data_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref, _ = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            
             # train the discriminator
            d_loss, d_losses_latent, signs_real = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            apa_stat.add(signs_real)

            d_loss, d_losses_ref, signs_real = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            apa_stat.add(signs_real)

            # computing moving average of losses
            add_loss_avg(d_losses_latent_avg, d_losses_latent, window_avg_len)
            add_loss_avg(d_losses_ref_avg, d_losses_ref, window_avg_len)
            add_loss_avg(g_losses_latent_avg, g_losses_latent, window_avg_len)
            add_loss_avg(g_losses_ref_avg, g_losses_ref, window_avg_len)

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # execute APA heuristic
            if args.use_apa and (i + 1) % args.apa_interval and i + 1 > args.apa_start:
                apa_stat.update()
                adjust = np.sign(apa_stat.mean() - args.apa_target) \
                         * (args.batch_size * args.apa_interval) / (args.apa_kimg * 1000)
                nets.discriminator.module.p.copy_((nets.discriminator.module.p + adjust).clamp_(0., args.apa_max_p))

            # print out log info
            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                for loss_avg, prefix in zip(
                        [d_losses_latent_avg, d_losses_ref_avg, g_losses_latent_avg, g_losses_ref_avg],
                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, avg in loss_avg.items():
                        all_losses[prefix + key] = avg.value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                log += f" loss/signs/real: {apa_stat.mean()}"
                log += f" augment: {nets.discriminator.module.p.cpu():.2f}"
                print(log)

            # generate images for debugging
            if (i + 1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i + 1)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)

            # compute FID and LPIPS if necessary
            if (i + 1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i + 1, mode='latent')
                calculate_metrics(nets_ema, args, i + 1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def sample_latent(self, loaders, target_domains, psi=1.0, show=False):
        args = self.args

        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))

        device = src.x.device
        N = src.x.size(0)

        # latent-guided image synthesis
        y_trg_list = [torch.tensor(y).repeat(N).to(device) for y in target_domains]
        z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)

        filename = ospj(args.result_dir, 'sample_latent_psi_%.1f.jpg' % (psi))
        results = utils.translate_using_latent_visual(self.nets_ema, src.x, y_trg_list, z_trg_list, psi)

        vutils.save_image(results, filename)

        if show:
            from matplotlib import pyplot as plt
            plt.imshow(results.transpose(0, 1).transpose(1, 2).numpy(), interpolation='nearest')
            plt.show()

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


class MovingAverage:

    def __init__(self, window=100):
        self.window = window
        self._values = []

    @property
    def value(self):
        return self._get_average()

    def add_value(self, val):
        if len(self._values) >= self.window:
            self._values = self._values[len(self._values) - self.window + 1:]

        self._values.append(val)

    def _get_average(self):
        if len(self._values) == 0:
            raise RuntimeError("Moving average with no values")

        sum = 0
        for v in self._values:
            sum += v

        return sum / len(self._values)


class StatCollector:

    def __init__(self, device):
        self._delta = torch.zeros([3], device=device, dtype=torch.float64)
        self.device = device
        self._cumulative = torch.zeros([3], device=device, dtype=torch.float64)

    def add(self, value: torch.Tensor):
        if value.numel() == 0:
            return

        elems: torch.Tensor = value.detach().flatten().to(torch.float32)

        moments = torch.stack([
            torch.ones_like(elems).sum(),
            elems.sum(),
            torch.pow(elems, 2).sum(),
        ]).to(torch.float64)

        self._cumulative.add_(moments)

    def update(self):
        self._delta.copy_(self._cumulative)
        self._cumulative = torch.zeros([3], device=self.device, dtype=torch.float64)

    def num(self):
        return int(self._delta[0])

    def mean(self):
        if int(self._delta[0]) == 0:
            return float("nan")
        return float(self._delta[1] / self._delta[0])

    def std(self):
        if int(self._delta[0]) == 0 or not np.isfinite(float(self._delta[1])):
            return float('nan')

        if int(self._delta[0]) == 1:
            return float(0)

        mean = float(self._delta[1] / self._delta[0])
        raw_var = float(self._delta[2] / self._delta[0])

        return np.sqrt(max(raw_var - np.square(mean), 0))


def add_loss_avg(loss_avg, loss, window_avg_len):
    if len(loss_avg) == 0:
        for k in loss.keys():
            loss_avg[k] = MovingAverage(window_avg_len)

    for k, v in loss.items():
        loss_avg[k].add_value(v)


def adaptive_pseudo_augmentation(p, real_img, pseudo_data):
    # Apply Adaptive Pseudo Augmentation (APA)
    batch_size = real_img.shape[0]
    pseudo_flag = torch.ones([batch_size, 1, 1, 1], device=real_img.device)
    pseudo_flag = torch.where(torch.rand([batch_size, 1, 1, 1], device=real_img.device) < p,
                              pseudo_flag, torch.zeros_like(pseudo_flag))
    if torch.allclose(pseudo_flag, torch.zeros_like(pseudo_flag)):
        return real_img
    else:
        assert pseudo_data is not None
        return pseudo_data * pseudo_flag + real_img * (1 - pseudo_flag)


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, pseudo_data=None):
    assert (z_trg is None) != (x_ref is None)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_org = nets.mapping_network(z_trg, y_org)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
            s_org = nets.style_encoder(x_ref, y_org)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
        pseudo_data = nets.generator(x_real, s_org, masks=masks)

    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    
    # APA augmentation
    if args.use_apa:
        x_real_augmented = adaptive_pseudo_augmentation(nets.discriminator.module.p, x_real, pseudo_data)
    else:
        x_real_augmented = x_real

    # with real images
    x_real.requires_grad_()
    x_real_augmented.requires_grad_()

    logits_real = nets.discriminator(x_real_augmented, y_org)
    loss_real = adv_loss(logits_real, 1)
    loss_reg = r1_reg(logits_real, x_real_augmented)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item()), logits_real.sign()


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
           - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item()), x_fake.detach()


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
