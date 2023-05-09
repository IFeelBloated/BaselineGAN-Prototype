import torch
import torch.nn as nn

class AdversarialTraining:
    def __init__(self, Generator, Discriminator):
        self.Generator = Generator
        self.Discriminator = Discriminator
        
    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])
        
    def AccumulateGeneratorGradients(self, Noise, RealSamples, Scale=1, Preprocessor=lambda x: x):
        FakeSamples = Preprocessor(self.Generator(Noise))
        RealSamples = Preprocessor(RealSamples).detach()
        
        FakeLogits = self.Discriminator(FakeSamples)
        RealLogits = self.Discriminator(RealSamples)
        
        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        (Scale * AdversarialLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]
    
    def AccumulateDiscriminatorGradients(self, Noise, RealSamples, Gamma, Scale=1, Preprocessor=lambda x: x):
        RealSamples = Preprocessor(RealSamples).detach().requires_grad_(True)
        FakeSamples = Preprocessor(self.Generator(Noise)).detach().requires_grad_(True)
        
        RealLogits = self.Discriminator(RealSamples)
        FakeLogits = self.Discriminator(FakeSamples)
        
        R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
        R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
        
        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
        (Scale * DiscriminatorLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]