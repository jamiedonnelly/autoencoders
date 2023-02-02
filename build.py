import inspect
import torch
from torch import nn
from torch.functional import F
import torch.fx as fx
import parsing

def _extract_shapes(encoder,input_shape):
        shapes = {}
        test = torch.randn(input_shape).unsqueeze(0)
        for layer in encoder:
            if isinstance(layer,nn.AdaptiveAvgPool2d):
                if 'Apool' in shapes.keys():
                    shapes['Apool'].append(test.shape)
                else:
                    shapes['Apool'] = [test.shape]
                test = layer(test)
            elif isinstance(layer,nn.MaxPool2d):
                if 'Mpool' in shapes.keys():
                    shapes['Mpool'].append(test.shape)
                else:
                    shapes['Mpool'] = [test.shape]
                test = layer(test)
            elif isinstance(layer,nn.Flatten):
                shapes['Flatten'] = test.shape[-3:]
                test = layer(test)
            else:
                test = layer(test)
        return shapes

def _create_decoder(encoder,shapes,input_shape):
    decoder = nn.Sequential()
    try:
        max_shape_iter = iter(reversed(shapes['Mpool']))
    except KeyError:
        pass
    try:
        a_shape_iter = iter(reversed(shapes['Apool']))
    except KeyError:
        pass
    for ix, layer in enumerate(reversed(encoder)):
        # Loop handles all convolution layers 
        if isinstance(layer,nn.Conv2d):
            conv_params = parsing._extract_params(layer)
            conv_params['in_channels'], conv_params['out_channels'] = conv_params['out_channels'], conv_params['in_channels']
            new_layer = nn.ConvTranspose2d(**conv_params)
            decoder.append(new_layer)
            for j in range(ix-1,0,-1):
                if isinstance(encoder[::-1][j],nn.Conv2d) or isinstance(encoder[::-1][j],nn.Linear) or\
                (isinstance(encoder[::-1][j],nn.MaxPool2d) or isinstance(encoder[::-1][j],nn.AdaptiveAvgPool2d)):
                    break
                elif isinstance(encoder[::-1][j],nn.BatchNorm2d):
                    bn_params = parsing._extract_params(encoder[::-1][j])
                    bn_params['num_features'] = conv_params['out_channels']
                    new_layer = type(encoder[::-1][j])(**bn_params)
                    decoder.append(new_layer)
                else:
                    new_layer = type(encoder[::-1][j])(**parsing._extract_params(encoder[::-1][j]))
                    decoder.append(new_layer)
            if ix+1 == len(encoder)-1:
                decoder.append(type(encoder[::-1][ix+1])(**parsing._extract_params(encoder[::-1][ix+1])))
        
        if isinstance(layer,nn.Flatten):
            new_layer = nn.Unflatten(1,shapes['Flatten'])
            decoder.append(new_layer)
        
        if isinstance(layer,nn.Linear):
            lin_params = parsing._extract_params(layer)
            lin_params['in_features'], lin_params['out_features'] = lin_params['out_features'], lin_params['in_features']
            new_layer = nn.Linear(**lin_params)
            decoder.append(new_layer)
            for j in range(ix-1,0,-1):
                if isinstance(encoder[::-1][j],nn.Conv2d) or isinstance(encoder[::-1][j],nn.Linear) or\
                (isinstance(encoder[::-1][j],nn.MaxPool2d) or isinstance(encoder[::-1][j],nn.AdaptiveAvgPool2d)):
                    break
                else:
                    new_layer = type(encoder[::-1][j])(**parsing._extract_params(encoder[::-1][j]))
                    decoder.append(new_layer)
        
        # Handles the maxpooling 
        if isinstance(layer,nn.MaxPool2d):
            shape = next(max_shape_iter)
            new_layer = nn.Upsample(size=(shape[-2],shape[-1]),mode='bilinear')
            decoder.append(new_layer)

        # Handles the adaptive-pooling
        if isinstance(layer,nn.AdaptiveAvgPool2d):
            shape = next(a_shape_iter)
            new_layer = nn.Upsample(size=(shape[-2],shape[-1]),mode='bilinear')
            decoder.append(new_layer)
            
        # During testing, certain models would return a shape of (C x (H-1), (W-1)) rather than 
        # (C x H x W) - diagnosing proved to be harder and a consistent solution was hard to obtain.
        # For instance, VGG-based models did not have this problem but Resnet did. 
        # Hacky solution is used to just upsample the final layer.
        try:
            encoder[::-1][ix+1]
        except IndexError:
            new_layer = nn.Upsample(size=(input_shape[-2],input_shape[-1]),mode='bilinear')
            decoder.append(new_layer)
        
        else:
            pass

    return decoder

def build_decoder(encoder,input_shape):
    shapes = _extract_shapes(encoder,input_shape)
    return _create_decoder(encoder,shapes,input_shape)

