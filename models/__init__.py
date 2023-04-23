from .vit import *

model_maps = {
  'vitstr_tiny_patch16_224': vit_t_16,
  'vitstr_small_patch16_224': vit_s_16,
  'vitstr_base_patch16_224': vit_b_16,
}


def create_vitstr(model_name: str, *args, **kwargs) -> VisionTransformer:
    return model_maps[model_name](*args, **kwargs)
