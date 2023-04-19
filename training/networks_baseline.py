import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .FusedOperators import BiasedActivation

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, Cardinality, ExpansionFactor, KernelSize):
        super(ResidualBlock, self).__init__()
        
        ExpandedChannels = InputChannels * ExpansionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, ExpandedChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Conv2d(ExpandedChannels, ExpandedChannels, kernel_size=KernelSize, stride=1, padding=(KernelSize - 1) // 2, groups=Cardinality, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(ExpandedChannels, InputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(ExpandedChannels)
        self.NonLinearity2 = BiasedActivation(ExpandedChannels)
        
    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return x + y
    
class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))
        
    def forward(self, x):
        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        x = self.Resampler(x)
        
        return x
    
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))
        
    def forward(self, x):
        x = self.Resampler(x)
        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        
        return x
    
class GenerativeBasis(nn.Module):
    def __init__(self, InputDimension, OutputChannels):
        super(GenerativeBasis, self).__init__()
        
        self.Basis = nn.Parameter(torch.empty(OutputChannels, 4, 4).normal_(0, 1))
        self.LinearLayer = MSRInitializer(nn.Linear(InputDimension, OutputChannels, bias=False))
        
    def forward(self, x):
        return self.Basis.view(1, -1, 4, 4) * self.LinearLayer(x).view(x.shape[0], -1, 1, 1)
    
class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels, OutputDimension):
        super(DiscriminativeBasis, self).__init__()
        
        self.Basis = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
        self.LinearLayer = MSRInitializer(nn.Linear(InputChannels, OutputDimension, bias=False))
        
    def forward(self, x):
        return self.LinearLayer(self.Basis(x).view(x.shape[0], -1))
    
class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, ResamplingFilter=None):
        super(GeneratorStage, self).__init__()
        
        TransitionLayer = GenerativeBasis(InputChannels, OutputChannels) if ResamplingFilter is None else UpsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        self.Layers = nn.ModuleList([TransitionLayer] + [ResidualBlock(OutputChannels, Cardinality, ExpansionFactor, KernelSize) for _ in range(NumberOfBlocks)])
        
    def forward(self, x):
        for Layer in self.Layers:
            x = Layer(x)
        
        return x
    
class DiscriminatorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, ResamplingFilter=None):
        super(DiscriminatorStage, self).__init__()
        
        TransitionLayer = DiscriminativeBasis(InputChannels, OutputChannels) if ResamplingFilter is None else DownsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        self.Layers = nn.ModuleList([ResidualBlock(InputChannels, Cardinality, ExpansionFactor, KernelSize) for _ in range(NumberOfBlocks)] + [TransitionLayer])
        
    def forward(self, x):
        for Layer in self.Layers:
            x = Layer(x)
        
        return x
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        MainLayers = [GeneratorStage(NoiseDimension, WidthPerStage[0], CardinalityPerStage[0], BlocksPerStage[0], ExpansionFactor, KernelSize)]
        MainLayers += [GeneratorStage(WidthPerStage[x], WidthPerStage[x + 1], CardinalityPerStage[x + 1], BlocksPerStage[x + 1], ExpansionFactor, KernelSize, ResamplingFilter) for x in range(len(WidthPerStage) - 1)]
        
        AggregationLayers = [MSRInitializer(nn.Conv2d(WidthPerStage[0], 3, kernel_size=1, stride=1, padding=0, bias=False))]
        AggregationLayers += [MSRInitializer(nn.Conv2d(WidthPerStage[x + 1], 3, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0) for x in range(len(WidthPerStage) - 1)]
        
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayers = nn.ModuleList(AggregationLayers)
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
        self.z_dim = NoiseDimension
        
    def forward(self, x):
        AggregatedOutput = None
        
        for Layer, Aggregate in zip(self.MainLayers, self.AggregationLayers):
            x = Layer(x)
            AggregatedOutput = self.Resampler(AggregatedOutput) + Aggregate(x) if AggregatedOutput is not None else Aggregate(x)
        
        return AggregatedOutput
    
class Discriminator(nn.Module):
    def __init__(self, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        MainLayers = [DiscriminatorStage(WidthPerStage[x], WidthPerStage[x + 1], CardinalityPerStage[x], BlocksPerStage[x], ExpansionFactor, KernelSize, ResamplingFilter) for x in range(len(WidthPerStage) - 1)]
        MainLayers += [DiscriminatorStage(WidthPerStage[-1], 1, CardinalityPerStage[-1], BlocksPerStage[-1], ExpansionFactor, KernelSize)]
        
        self.ExtractionLayer = MSRInitializer(nn.Conv2d(3, WidthPerStage[0], kernel_size=1, stride=1, padding=0, bias=False))
        self.MainLayers = nn.ModuleList(MainLayers)
        
    def forward(self, x):
        x = self.ExtractionLayer(x)
        
        for Layer in self.MainLayers:
            x = Layer(x)
        
        return x.view(x.shape[0])