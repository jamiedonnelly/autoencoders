# Base imports
import torch 
from torch import nn
from typing import List, Tuple

# Module imports    
from models import model_dict
import decoder 
import parsing

def run_encoder_test(model: nn.Module, skip_layers: List = [], input_shape: Tuple = (3,224,224)) -> nn.Sequential:
    """_summary_

    Args:
        model (nn.Module): _description_
    """

    # Parse model into `nn.Sequential` wrapped
    parsed_model = parsing._parse_torchvision_model(model,skip_layers=skip_layers)
    
    # Forward pass 
    model_output = model.forward(torch.randn(input_shape).unsqueeze(0))
    parsed_model_output = parsed_model(torch.randn(input_shape).unsqueeze(0))
   
    # Test
    assert model_output.shape == parsed_model_output.shape, f"Expected output shape {model_output.shape} got shape\
         {parsed_model_output.shape}"

    # Test passed
    print("Forward pass test successful...")

    return parsed_model


def run_decoder_test(parsed_model: nn.Sequential, input_shape: Tuple = (3,224,224)) -> None:
    """_summary_

    Args:
        parse_model (nn.Sequential): _description_
        input_shape (Tuple, optional): _description_. Defaults to (3,224,224).
    """

    # Run forward pass of parsed model to get output shape
    encoded = parsed_model(torch.randn(input_shape).unsqueeze(0))
    encoded_shape = encoded.shape

    # Construct decoder model 
    conv_decoder = decoder.ConvolutionalDecoder(parsed_model, input_shape)

    # Run decoding pass
    decoded = conv_decoder(torch.randn(encoded_shape))
    decoded_shape = decoded.squeeze(0).shape

    # Test
    assert decoded_shape == input_shape, f"Expected output shape {input_shape} got shape {decoded_shape}"
    
    # Test passed 
    print("Decoded test successful...")


if __name__=="__main__":

    # Default input is RGB 224 x 224 image
    default_input_shape = (3,224,224)

    for key in model_dict.keys():

        print(f"\nTesting...{model_dict[key].__name__}")

        # Initialise skip_layers - empty for all except ResNet models
        skip_layers = []

        # Skip resnet downsampling layers
        if 'res' in key:
            skip_layers.append('downsample')

        # Instantiate torchvision model from `model_dict`
        model = model_dict[key]() 

        # Run tests
        parsed_model = run_encoder_test(model,skip_layers,default_input_shape)
        run_decoder_test(parsed_model,default_input_shape)
    