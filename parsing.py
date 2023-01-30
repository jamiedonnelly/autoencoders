import inspect
from torch import nn
import torch.fx as fx

# Used to extract param dict from layer instances 
# and can be used to construct new layers from existing layers.
# New layers are used as it's easier to modify new layers before construction than
# to modify existing layers with initialised weights and biases.
def _extract_params(layer):
    params = {}
    signature_params = inspect.signature(type(layer)).parameters
    for key in signature_params.keys():
        if key == 'bias':
            if layer.__getattr__(key) == None:
                params[key]=False
            else:
                params[key]=True
        else:
            try:
                params[key] = layer.__getattr__(signature_params[key].name) 
            except AttributeError:
                try:
                    params[key] = layer.__getattribute__(signature_params[key].name)
                except:
                    break
    return params

# Function to check if a linear layer exists in the parsed model 
# used to determine where and when to add a flatten layer to move from 
# convolutions to dense mappings
def _check_linear(model):
    return [layer for layer in model if isinstance(layer,nn.Linear)]

# Function to be used in `_parse_torchvision_model` to determine if a string belongs in a set of layers to be skipped
# i.e., to determin if `'downsample'` is in a set of keys such as, `['layer1.downsample.bn',...,'layerN.downsample.relu']`
def _check_substring(strings,target):
    return [True for string in strings if string in target]

# Function to parse the torchvision model structure.
# Will not parse any arbitrary structure because it assumes nesting doesn't occur 
# but will appropriately parse models in the `torchvision.models` module.
# Can take a list of keywords/types of layers to skip e.g., `nn.BatchNorm2d` or `'downsample'`.
def _parse_torchvision_model(model,skip_layers=[]):
    new_model = nn.Sequential()
    trace = fx.symbolic_trace(model)
    modules = dict(trace.named_modules())
    for key in modules.keys():
        if len(modules[key].__getattribute__('_modules')) > 0:
            pass
        else:
            if (_check_substring(skip_layers,key)) or (type(modules[key]) in skip_layers):
                pass
            else:
                if (type(modules[key])==nn.Linear) and (not _check_linear(new_model)):
                    new_model.append(nn.Flatten(1))
                new_model.append(type(modules[key])(**_extract_params(modules[key])))
    return new_model
