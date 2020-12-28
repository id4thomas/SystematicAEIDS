import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self,model_config):
        super(AE, self).__init__()
        self.data_dim = model_config['d_dim']
        layer_dims = model_config['layers']
        
        num_layers=len(layer_dims)
        self.encoder = nn.Sequential()
        self.encoder.add_module("e_in",nn.Linear(self.data_dim,layer_dims[0]))
        self.encoder.add_module("e_in_actv".format(1),nn.LeakyReLU())
                
        for i in range(num_layers-1):
            self.encoder.add_module("e_{}".format(i+1),nn.Linear(layer_dims[i],layer_dims[i+1]))
            self.encoder.add_module("e_{}_actv".format(i+1),nn.LeakyReLU())
            
        layer_dims.reverse()
        self.decoder = nn.Sequential()
        for i in range(num_layers-1):
            # print(layer_dims[i],layer_dims[i+1])
            self.decoder.add_module("d_{}".format(i+1),nn.Linear(layer_dims[i],layer_dims[i+1]))
            self.decoder.add_module("d_{}_actv".format(i+1),nn.LeakyReLU())
        self.decoder.add_module("d_out".format(i+1),nn.Linear(layer_dims[num_layers-1],self.data_dim))
            
    def forward(self,x):
        f = self.encoder(x)
        output = self.decoder(f)
        return {'output': output}
    
    def compute_loss(self, outputs, target):
        output = outputs['output']
        loss = torch.nn.MSELoss()(output, target)
        return loss