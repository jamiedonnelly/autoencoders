from torch import nn
import math

class Autoencoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(Autoencoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._init()

    def _init(self):
        layers = [l for l in self.encoder]
        layers.extend([l for l in self.decoder])
        for layer in layers:
            # weight init
            try:
                weight = layer.weight
                nn.init.kaiming_uniform_(weight,math.sqrt(5,0))
            except:
                pass
            # bias init
            try:
                bias = layer.bias
                nn.init.uniform_(bias)
            except:
                pass

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)




