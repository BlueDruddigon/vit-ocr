import argparse

import torch

from model_v2 import Model
from models import create_vitstr

weights = '/home/beosup/Documents/vitstr/pretrained/vitstr_base_patch16_224_aug.pth'
pt_weights = '/home/beosup/Documents/vitstr/pretrained/vitstr_small_patch16_jit.pt'

state_dict = torch.load(weights)
state_dict = {k.replace('module.vitstr.', ''): v for k, v in state_dict.items()}
vit = create_vitstr('vitstr_base_patch16_224', weights=None, num_classes=96).cuda()
vit.load_state_dict(vit.custom_load_state_dict(state_dict))
torch.save(vit.state_dict(), './pretrained/vit_b_16.pth')

parser = argparse.ArgumentParser()
parser.add_argument('--transformer', default='vitstr_base_patch16_224')
parser.add_argument('--num_class', default=96)

opt = parser.parse_args()

ckpt = torch.load(weights)
ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
m = Model(opt).cuda()
m.load_state_dict(ckpt)

zi = torch.ones((1, 1, 224, 224)).cuda()
with torch.no_grad():
    o2 = m(zi)
with torch.no_grad():
    o1 = vit(zi)

# print(o1, o1.shape)
# print(o2, o2.shape)

mo = torch.jit.load(pt_weights)
mo = mo.cuda()
with torch.no_grad():
    o3 = mo(zi)

print(o1, o1.shape)
print(o2, o2.shape)
print(o3, o3.shape)
