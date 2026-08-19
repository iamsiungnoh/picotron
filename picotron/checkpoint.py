import contextlib
import os

import torch
import torch.nn as nn

import picotron.process_group_manager as pgm
from picotron.utils import assert_no_meta_tensors



@contextlib.contextmanager
def init_model_with_dematerialized_weights(include_buffers: bool = False):
    """Construct a model with parameters on the meta device."""
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, parameter):
        old_register_parameter(module, name, parameter)
        if parameter is not None:
            parameter_class = type(module._parameters[name])
            parameter_attributes = module._parameters[name].__dict__
            module._parameters[name] = parameter_class(
                module._parameters[name].to(torch.device("meta")),
                **parameter_attributes,
            )

    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer


def init_model_with_materialized_weights(model, device="cpu"):
    """Allocate meta parameters directly and initialize them from scratch."""
    for module in model.modules():
        for name, parameter in module._parameters.items():
            if parameter is None or not parameter.is_meta:
                continue

            materialized_parameter = nn.Parameter(
                torch.empty_like(parameter, device=device),
                requires_grad=parameter.requires_grad,
            )
            materialized_parameter.__dict__.update(parameter.__dict__)
            module._parameters[name] = materialized_parameter

    model.reset_parameters()
    assert_no_meta_tensors(model)
    return model

class InitializationManager:
    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config

    def init_model_parameters(self):
        self.model.reset_parameters()

    def get_layer_names_in_sft_format(self):
        """Get layer names in safetensors format based on model's layer distribution."""
        decoder_components = [
            "input_layernorm",
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "post_attention_layernorm",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.v_proj",
        ]
        
        # Generate base layer names
        layer_names = []
        if isinstance(self.model, PipelineParallel):
            base_names = [f"model.layers.{id}" for id in self.model.layer_distribution]
        else:
            base_names = [f"model.layers.{id}" for id in range(self.model_config.num_hidden_layers)]
        
        for layer in base_names:
            for component in decoder_components:
                layer_names.append(f"{layer}.{component}.weight")
       
        # Add special layers based on pipeline stage or non-PP case
        # NOTE: Safetensors may have tied embeddings, but Picotron does not support it. We always create a new lm_head.
        if isinstance(self.model, PipelineParallel):
            if pgm.process_group_manager.pp_is_first_stage:
                layer_names.insert(0, "model.embed_tokens.weight")
            elif pgm.process_group_manager.pp_is_last_stage:
                layer_names.extend(["model.norm.weight"])
        else:
            layer_names.insert(0, "model.embed_tokens.weight")
            layer_names.extend(["model.norm.weight"])

        return layer_names

    def adjust_tensor_size(self, tensor, name):
        """Resize tensor based on architecture changes and tensor parallelism."""
        tp_rank = pgm.process_group_manager.tp_rank
        tp_size = pgm.process_group_manager.tp_world_size
        hidden_size = self.model_config.hidden_size
        
        # Handle embedding and final projection layers
        if 'embedding.weight' in name or 'final_proj.weight' in name:
            vocab_size = self.model_config.vocab_size

            if self.model_config.vocab_padding_en:
                padded_vocab_size = (
                    (vocab_size + tp_size - 1)
                    // tp_size
                    * tp_size
                )
                vocab_per_rank = padded_vocab_size // tp_size
            else:
                vocab_per_rank = vocab_size // tp_size

            start_idx = tp_rank * vocab_per_rank
            end_idx = start_idx + vocab_per_rank

            if tensor.shape[0] != vocab_per_rank:
                tensor = tensor[start_idx:end_idx, :]

            if tensor.shape[0] < vocab_per_rank:
                pad_rows = vocab_per_rank - tensor.shape[0]

                pad_tensor = torch.zeros(
                    pad_rows,
                    tensor.shape[1],
                    dtype=tensor.dtype,
                    device=tensor.device,
                )

                tensor = torch.cat(
                    [tensor, pad_tensor],
                    dim=0,
                )
            return tensor

        # Handle attention layers
        if 'attention' in name:
            head_dim = hidden_size // self.model_config.num_attention_heads
            
            if 'q_proj.weight' in name:
                total_heads = self.model_config.num_attention_heads
                heads_per_rank = total_heads // tp_size
                target_dim = heads_per_rank * head_dim
            elif 'k_proj.weight' in name or 'v_proj.weight' in name:
                total_heads = self.model_config.num_key_value_heads
                heads_per_rank = total_heads // tp_size
                target_dim = heads_per_rank * head_dim
            elif 'out_proj.weight' in name:
                # For out_proj, we split along the second dimension
                target_dim = tensor.shape[0]  # First dimension stays the same
                if tensor.shape[1] != hidden_size // tp_size:
                    tensor = tensor[:, (hidden_size // tp_size) * tp_rank:(hidden_size // tp_size) * (tp_rank + 1)]
                return tensor
            else:
                return tensor
                
            if tensor.shape[0] != target_dim:
                if target_dim > tensor.shape[0]:
                    pad_tensor = torch.empty(target_dim - tensor.shape[0], tensor.shape[1], 
                                        dtype=tensor.dtype, device=tensor.device)
                    tensor = torch.cat([tensor, pad_tensor], dim=0)
                else:
                    start_idx = tp_rank * target_dim
                    end_idx = start_idx + target_dim
                    tensor = tensor[start_idx:end_idx, :]
                    tensor = tensor[:target_dim, :]

        # Handle MLP layers
        elif 'mlp' in name:
            intermediate_size = self.model_config.intermediate_size
            intermediate_size_per_rank = intermediate_size // tp_size
            
            if 'up_proj.weight' in name or 'gate_proj.weight' in name:
                if tensor.shape[0] != intermediate_size_per_rank:
                    start_idx = tp_rank * intermediate_size_per_rank
                    end_idx = start_idx + intermediate_size_per_rank
                    tensor = tensor[start_idx:end_idx, :]
            elif 'down_proj.weight' in name:
                if tensor.shape[1] != intermediate_size_per_rank:
                    start_idx = tp_rank * intermediate_size_per_rank
                    end_idx = start_idx + intermediate_size_per_rank
                    tensor = tensor[:, start_idx:end_idx]
                    
        return tensor

    def convert_safetensors_to_hf_name(self, sft_name):
        """Convert safetensors naming convention to HuggingFace naming convention."""
        name_mapping = {
            "model.": "",
            "layers.": "decoder_layers.",
            "embed_tokens": "embedding",
            "self_attn.": "attention.",
            "o_proj": "out_proj",
            "lm_head": "final_proj",
            "input_layernorm": "input_layernorm",
            "post_attention_layernorm": "post_attention_layernorm",
            r'^norm': 'final_norm'
        }
        
        result = sft_name
        for pattern, replacement in name_mapping.items():
            result = re.sub(pattern, replacement, result)
        return result

class CheckpointManager:
    def __init__(self):
        self.tp_rank = pgm.process_group_manager.tp_rank
        self.pp_rank = pgm.process_group_manager.pp_rank
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.pp_world_size = pgm.process_group_manager.pp_world_size
        self.cp_dp_world_size = pgm.process_group_manager.cp_dp_world_size
        self.dp_rank = pgm.process_group_manager.dp_rank
        self.cp_rank = pgm.process_group_manager.cp_rank

    def _get_checkpoint_path(self, out_dir):
        checkpoint_name = (
            f"weights_tp_rank_world_size={self.tp_rank}_{self.tp_world_size}_"
            f"pp_rank_world_size={self.pp_rank}_{self.pp_world_size}.pth"
        )
        return os.path.join(out_dir, checkpoint_name)

    def save_checkpoint(self, model, optimizer, trained_steps, trained_tokens, out_dir):
        """Save model, optimizer, and training progress."""
        path = self._get_checkpoint_path(out_dir)

        # CP/DP replicas have identical weights, so only replica zero saves.
        if self.dp_rank == 0 and self.cp_rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            raw_model = model.module if self.cp_dp_world_size > 1 else model
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "trained_steps": trained_steps,
                "trained_tokens": trained_tokens,
            }
            torch.save(checkpoint, path)

    def load_checkpoint(self, model, optimizer, out_dir):
        """Load a checkpoint created with the same parallel topology."""
        path = self._get_checkpoint_path(out_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path)
        raw_model = model.module if self.cp_dp_world_size > 1 else model
        raw_model.load_state_dict(checkpoint['model'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        return checkpoint['trained_steps'], checkpoint['trained_tokens']


def _fuse_qkv_state_dict(state_dict, model_config):
    layer_indices = set()
    for key in state_dict.keys():
        m = re.match(r"decoder_layers\.(\d+)\.attention\.q_proj\.weight", key)
        if m:
            layer_indices.add(int(m.group(1)))

    for idx in layer_indices:
        prefix = f"decoder_layers.{idx}.attention"
        q_key, k_key, v_key = f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"
        # NEW: pop (not just read) so the separate keys don't remain as "unexpected keys"
        q_w = state_dict.pop(q_key)
        k_w = state_dict.pop(k_key)
        v_w = state_dict.pop(v_key)
        state_dict[f"{prefix}.qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

    return state_dict
