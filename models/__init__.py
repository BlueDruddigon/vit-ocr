from .vit import VisionTransformer, vit_b_16, vit_s_16, vit_t_16

model_maps = {
  'vit_t_16': vit_t_16,
  'vit_s_16': vit_s_16,
  'vit_b_16': vit_b_16,
}


def create_vitstr(model_name: str, *args, **kwargs) -> VisionTransformer:
    return model_maps[model_name](*args, **kwargs)
