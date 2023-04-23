import torch

from models import create_vitstr

weights = '/home/beosup/Documents/vitstr/pretrained/vitstr_tiny_patch16_224_aug.pth'

state_dict = torch.load(weights)
state_dict = {k.replace('module.vitstr.', ''): v for k, v in state_dict.items()}
vit = create_vitstr('vit_t_16', weights=None, num_classes=96).cuda()
vit.load_state_dict(vit.custom_load_state_dict(state_dict))
ckpt = vit.state_dict()
ckpt = { f'module.vitstr.{k}': v for k, v in ckpt.items() }
torch.save(vit.state_dict(), './pretrained/vit_t_16.pth')
