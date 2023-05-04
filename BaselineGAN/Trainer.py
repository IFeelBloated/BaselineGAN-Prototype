import torch
import torch.nn as nn

class AdversarialTraining:
    def __init__(self, Generator, Discriminator, Laziness=0):
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.Laziness = Laziness
        
        assert Laziness % 2 == 0
        DiscriminatorGradientAccumulator = AdversarialTraining.AccumulateDiscriminatorGradientsWithEagerRegularization if Laziness == 0 else AdversarialTraining.AccumulateDiscriminatorGradientsWithLazyRegularization
        
        self.AccumulateDiscriminatorGradients = lambda *Arguments, **NamedArguments: DiscriminatorGradientAccumulator(self, *Arguments, **NamedArguments)
        
    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])
        
    def AccumulateGeneratorGradients(self, Noise, RealSamples, Scale=1):
        FakeSamples = self.Generator(Noise)
        
        FakeLogits = self.Discriminator(FakeSamples)
        RealLogits = self.Discriminator(RealSamples.detach())
        
        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        (Scale * AdversarialLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]
    
    @staticmethod
    def AccumulateDiscriminatorGradientsWithEagerRegularization(self, Noise, RealSamples, Gamma, Scale=1):
        RealSamples = RealSamples.detach().requires_grad_(True)
        FakeSamples = self.Generator(Noise).detach().requires_grad_(True)
        
        RealLogits = self.Discriminator(RealSamples)
        FakeLogits = self.Discriminator(FakeSamples)
        
        R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
        R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
        
        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
        (Scale * DiscriminatorLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]
    
    @staticmethod
    def AccumulateDiscriminatorGradientsWithLazyRegularization(self, Noise, RealSamples, BatchIndex, Gamma, Scale=1):
        RealSamples = RealSamples.detach().requires_grad_(True)
        FakeSamples = self.Generator(Noise).detach().requires_grad_(True)
        
        RealLogits = self.Discriminator(RealSamples)
        FakeLogits = self.Discriminator(FakeSamples)
        
        R1Penalty = None
        R2Penalty = None
        
        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        DiscriminatorLoss = AdversarialLoss
        
        if BatchIndex % self.Laziness == 0:
            R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
            DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * self.Laziness * R1Penalty
            
        if BatchIndex % self.Laziness == self.Laziness // 2:
            R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
            DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * self.Laziness * R2Penalty
            
        (Scale * DiscriminatorLoss.mean()).backward()
        
        return [x.detach() if x is not None else None for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]