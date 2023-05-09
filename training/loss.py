# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
import numpy as np
import torch
from BaselineGAN.Trainer import AdversarialTraining

#----------------------------------------------------------------------------

class BaselineGANLoss:
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        self.r1_gamma = r1_gamma
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.trainer = AdversarialTraining(G, D)
        
    @staticmethod
    def create_preprocessor(blur_sigma):
        blur_size = np.floor(blur_sigma * 3)
        
        if blur_size > 0:
            def preprocessor(img):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                return upfirdn2d.filter2d(img, f / f.sum())
            return preprocessor
        
        return lambda x: x

    def accumulate_gradients(self, phase, real_img, gen_z, gain, cur_nimg):
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        training_stats.report('Loss/blur_sigma', blur_sigma)
        
        # G
        if phase == 'G':
            AdversarialLoss, RelativisticLogits = self.trainer.AccumulateGeneratorGradients(gen_z, real_img, gain, BaselineGANLoss.create_preprocessor(blur_sigma))
            
            training_stats.report('Loss/scores/fake', RelativisticLogits)
            training_stats.report('Loss/signs/fake', RelativisticLogits.sign())
            training_stats.report('Loss/G/loss', AdversarialLoss)
            
        # D
        if phase == 'D':
            AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty = self.trainer.AccumulateDiscriminatorGradients(gen_z, real_img, self.r1_gamma, gain, BaselineGANLoss.create_preprocessor(blur_sigma))
            
            training_stats.report('Loss/scores/real', RelativisticLogits)
            training_stats.report('Loss/signs/real', RelativisticLogits.sign())
            training_stats.report('Loss/D/loss', AdversarialLoss)
            training_stats.report('Loss/r1_penalty', R1Penalty)
            training_stats.report('Loss/r2_penalty', R2Penalty)
#----------------------------------------------------------------------------