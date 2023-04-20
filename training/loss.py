# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------

class StyleGAN2Loss():
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.r2_gamma           = self.r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

    def run_G(self, z, update_emas=False):
        return self.G(z)

    def run_D(self, img, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        return self.D(img)

    def accumulate_gradients(self, phase, real_img, gen_z, gain, cur_nimg):
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # G
        if phase == 'G':
            with torch.autograd.profiler.record_function('G_forward'):
                gen_img = self.run_G(gen_z)
                gen_logits = self.run_D(gen_img, blur_sigma=blur_sigma)
                real_logits = self.run_D(real_img.detach(), blur_sigma=blur_sigma)
                relativistic_logits = gen_logits - real_logits
                
                training_stats.report('Loss/scores/fake', relativistic_logits)
                training_stats.report('Loss/signs/fake', relativistic_logits.sign())
                
                loss_Gmain = torch.nn.functional.softplus(-relativistic_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
                
            with torch.autograd.profiler.record_function('G_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # D
        if phase == 'D':
            with torch.autograd.profiler.record_function('D_forward'):
                gen_img = self.run_G(gen_z)
                gen_img_tmp = gen_img.detach().requires_grad_(True)
                gen_logits = self.run_D(gen_img_tmp, blur_sigma=blur_sigma)
                
                real_img_tmp = real_img.detach().requires_grad_(True)
                real_logits = self.run_D(real_img_tmp, blur_sigma=blur_sigma)
                
                relativistic_logits = real_logits - gen_logits
                
                training_stats.report('Loss/scores/real', relativistic_logits)
                training_stats.report('Loss/signs/real', relativistic_logits.sign())
                
                loss_Dmain = torch.nn.functional.softplus(-relativistic_logits)
                training_stats.report('Loss/D/loss', loss_Dmain)
                
                with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().sum([1,2,3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                training_stats.report('Loss/r1_penalty', r1_penalty)
                training_stats.report('Loss/D/r1_reg', loss_Dr1)
                
                with torch.autograd.profiler.record_function('r2_grads'), conv2d_gradfix.no_weight_gradients():
                    r2_grads = torch.autograd.grad(outputs=[gen_logits.sum()], inputs=[gen_img_tmp], create_graph=True, only_inputs=True)[0] 
                r2_penalty = r2_grads.square().sum([1,2,3])
                loss_Dr2 = r2_penalty * (self.r2_gamma / 2)
                training_stats.report('Loss/r2_penalty', r2_penalty)
                training_stats.report('Loss/D/r2_reg', loss_Dr2)
                
                loss_D = loss_Dmain + loss_Dr1 + loss_Dr2
                
            with torch.autograd.profiler.record_function('D_backward'):
                loss_D.mean().mul(gain).backward()
                
                
#----------------------------------------------------------------------------
