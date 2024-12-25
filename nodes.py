from abc import ABC
import comfy.utils
import folder_paths as comfy_paths
import os

up_layer_names = ["lora_up", "lora.up", "lora_A", "lora_linear_layer.up"]
down_layer_names = ["lora_down", "lora.down", "lora_B", "lora_linear_layer.down"]
mid_layer_names = ["lora_mid"]

class _Smooth_Step_Lora_Loader_Base(ABC):

    def __init__(self):
        self.loaded_lora = None
        self.strength_smooth_step = 1.0

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, strength_smooth_step, **kwargs):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        if model is None:
            return (model, clip)

        lora_path = comfy_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None or strength_smooth_step != self.strength_smooth_step:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

            if strength_smooth_step > 0.0:
                lora = self.smooth_step_lora(lora, strength_smooth_step, **kwargs)
            self.strength_smooth_step = strength_smooth_step

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora, os.path.splitext(os.path.basename(lora_name))[0])
    
    def gather_kv(self, sd, **kwargs):
        pass

    def smooth_step_lora(self, sd, factor, **kwargs):
        keys_to_normalize, values_to_normalize = self.gather_kv(sd, **kwargs)

        min_value = 1.0
        max_value = 0.0
        for key, value in zip(keys_to_normalize, values_to_normalize):
            if value.min() < min_value:
                min_value = value.min()
            if value.max() > max_value:
                max_value = value.max()

        for key, value in zip(keys_to_normalize, values_to_normalize):
            min_val = min_value
            max_val = max_value
            normalized_value = (value - min_val) / (max_val - min_val + 1e-7) 
            adjusted_value = self.smooth_step_function(normalized_value)
            adjusted_value = min_val + adjusted_value * (max_val - min_val)
            adjusted_value = value * (1 - factor) + adjusted_value * factor
            sd[key] = adjusted_value

        return sd

    def smooth_step_function(self, x):
        return 3 * x*x - 2 * x * x* x


class Smooth_Step_Lora_Loader(_Smooth_Step_Lora_Loader_Base):
        
    @classmethod
    def INPUT_TYPES(s):
        file_list = comfy_paths.get_filename_list("loras")
        file_list.insert(0, "None")
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (file_list, ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_smooth_step": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 11.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, strength_smooth_step):
        return super().load_lora(model, clip, lora_name, strength_model, strength_clip, strength_smooth_step)

    def gather_kv(self, sd, *args):
        keys_to_normalize = []
        values_to_normalize = []
        for key in sd.keys():
            for layer in [*down_layer_names, *up_layer_names, *mid_layer_names]:
                if layer in key:
                    keys_to_normalize.append(key)
                    values_to_normalize.append(sd[key])
        return keys_to_normalize, values_to_normalize

NODE_CLASS_MAPPINGS = {
    "Smooth_Step_Lora_Loader": Smooth_Step_Lora_Loader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Smooth_Step_Lora_Loader": "Smooth Step Lora Loader"
}