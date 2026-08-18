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
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint["trained_steps"], checkpoint["trained_tokens"]
