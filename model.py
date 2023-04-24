import torch
import torch.nn as nn

from models import create_vitstr


class Model(nn.Module):
    def __init__(self, opts):
        super(Model, self).__init__()
        
        self.opts = opts
        if self.opts.transformation == 'TPS':
            self.transformation = nn.Linear()
        else:
            print('No Transformation module specified')
            self.transformation = None
        
        self.vit = create_vitstr(model_name=self.opts.transformer, num_classes=self.opts.num_classes)
    
    def forward(self, inputs: torch.Tensor, seqlen: int = 25):
        if self.transformation is not None:
            inputs = self.transformation(inputs)
        
        prediction = self.vitstr(inputs, seqlen=seqlen)
        return prediction


class JitModel(Model):
    def __init__(self, opts):
        super(JitModel, self).__init__(opts)
        self.vit = create_vitstr(model_name=opts.transformer, num_classes=opts.num_classes)
    
    def forward(self, inputs):
        return self.vit(inputs)
