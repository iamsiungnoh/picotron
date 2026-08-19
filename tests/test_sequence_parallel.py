"""
Run with:
PYTHONPATH=. CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nproc-per-node=2 tests/test_sequence_parallel.py
"""

import datetime
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from picotron.process_group_manager import setup_process_group_manager
from picotron.model import Llama
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel
from picotron.tensor_parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from picotron.tensor_parallel.tp_communications import (
    GatherFromSequenceParallelRegion,
    ReduceScatterToSequenceParallelRegion,
    ScatterToSequenceParallelRegion,
)


local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device("cuda", local_rank)

dist.init_process_group(
    rank=global_rank,
    world_size=world_size,
    backend="nccl",
    init_method="env://",
    timeout=datetime.timedelta(minutes=3),
)
setup_process_group_manager(tp_size=world_size, cp_size=1, pp_size=1, dp_size=1)
os.environ["CONTEXT_PARALLEL"] = "0"
os.environ["DEVICE"] = "cuda"
os.environ["DTYPE"] = "float32"
os.environ["FLASH_ATTEN"] = "0"

batch_size, local_sequence_length, hidden_size = 2, 2, 3

# Scatter forward, all-gather backward.
replicated_sequence = torch.arange(
    batch_size * local_sequence_length * world_size * hidden_size,
    dtype=torch.float32,
    device=device,
).view(batch_size, local_sequence_length * world_size, hidden_size).requires_grad_(True)
scattered = ScatterToSequenceParallelRegion.apply(replicated_sequence)
torch.testing.assert_close(
    scattered,
    replicated_sequence.chunk(world_size, dim=1)[global_rank],
)
scattered.sum().backward()
torch.testing.assert_close(replicated_sequence.grad, torch.ones_like(replicated_sequence))

# All-gather forward, reduce-scatter backward.
local_shard = torch.full(
    (batch_size, local_sequence_length, hidden_size),
    global_rank + 1,
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)
gathered = GatherFromSequenceParallelRegion.apply(local_shard)
expected_gathered = torch.cat(
    [torch.full_like(local_shard, rank + 1) for rank in range(world_size)],
    dim=1,
)
torch.testing.assert_close(gathered, expected_gathered)

gathered.sum().backward()
torch.testing.assert_close(local_shard.grad, torch.full_like(local_shard, world_size))

# Reduce-scatter forward, all-gather backward.
full_sequence_length = local_sequence_length * world_size
partial_output = torch.full(
    (batch_size, full_sequence_length, hidden_size),
    global_rank + 1,
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)
sequence_shard = ReduceScatterToSequenceParallelRegion.apply(partial_output)
expected_value = world_size * (world_size + 1) / 2
torch.testing.assert_close(
    sequence_shard,
    torch.full_like(sequence_shard, expected_value),
)

sequence_shard.sum().backward()
torch.testing.assert_close(partial_output.grad, torch.ones_like(partial_output))

# Column-parallel gather followed by row-parallel reduce-scatter.
input_size, intermediate_size, output_size = 4, 8, 4
torch.manual_seed(42)
reference_input = torch.randn(
    batch_size,
    full_sequence_length,
    input_size,
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)
reference_column = torch.nn.Linear(input_size, intermediate_size, bias=True, device=device)
reference_row = torch.nn.Linear(intermediate_size, output_size, bias=True, device=device)

sp_input = reference_input.detach().chunk(world_size, dim=1)[global_rank].contiguous().requires_grad_(True)
sp_column = ColumnParallelLinear(
    input_size,
    intermediate_size,
    bias=True,
    sequence_parallel=True,
).to(device)
sp_row = RowParallelLinear(
    intermediate_size,
    output_size,
    bias=True,
    sequence_parallel=True,
).to(device)

with torch.no_grad():
    sp_column.weight.copy_(reference_column.weight.chunk(world_size, dim=0)[global_rank])
    sp_column.bias.copy_(reference_column.bias.chunk(world_size, dim=0)[global_rank])
    sp_row.weight.copy_(reference_row.weight.chunk(world_size, dim=1)[global_rank])
    sp_row.bias.copy_(reference_row.bias)

reference_output = reference_row(reference_column(reference_input))
sp_output = sp_row(sp_column(sp_input))
torch.testing.assert_close(
    sp_output,
    reference_output.chunk(world_size, dim=1)[global_rank],
)

reference_output.sum().backward()
sp_output.sum().backward()
torch.testing.assert_close(
    sp_input.grad,
    reference_input.grad.chunk(world_size, dim=1)[global_rank],
)
torch.testing.assert_close(
    sp_column.weight.grad,
    reference_column.weight.grad.chunk(world_size, dim=0)[global_rank],
)
torch.testing.assert_close(
    sp_column.bias.grad,
    reference_column.bias.grad.chunk(world_size, dim=0)[global_rank],
)
torch.testing.assert_close(
    sp_row.weight.grad,
    reference_row.weight.grad.chunk(world_size, dim=1)[global_rank],
)
torch.testing.assert_close(sp_row.bias.grad, reference_row.bias.grad)

# End-to-end model boundary and replicated RMSNorm-gradient parity.
model_config = SimpleNamespace(
    hidden_size=8,
    intermediate_size=16,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=1,
    vocab_size=16,
    max_position_embeddings=full_sequence_length,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)
torch.manual_seed(123)
tp_model = apply_tensor_parallel(Llama(model_config), sequence_parallel=False).to(device)
sp_model = apply_tensor_parallel(Llama(model_config), sequence_parallel=True).to(device)
sp_model.load_state_dict(tp_model.state_dict())

input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)
tp_logits = tp_model(input_ids)
sp_logits = sp_model(input_ids)
torch.testing.assert_close(sp_logits, tp_logits, rtol=1e-5, atol=1e-5)

tp_logits.sum().backward()
sp_logits.sum().backward()
for (tp_name, tp_parameter), (sp_name, sp_parameter) in zip(
    tp_model.named_parameters(),
    sp_model.named_parameters(),
):
    assert tp_name == sp_name
    torch.testing.assert_close(
        sp_parameter.grad,
        tp_parameter.grad,
        rtol=1e-4,
        atol=1e-4,
        msg=lambda message: f"Gradient mismatch for {tp_name}: {message}",
    )

# Repeat model parity through the production bfloat16 Flash Attention and
# Triton RMSNorm path.
os.environ["DTYPE"] = "bfloat16"
os.environ["FLASH_ATTEN"] = "1"
flash_model_config = SimpleNamespace(
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=1,
    vocab_size=16,
    max_position_embeddings=full_sequence_length,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)
torch.manual_seed(456)
tp_flash_model = apply_tensor_parallel(
    Llama(flash_model_config),
    sequence_parallel=False,
).to(device=device, dtype=torch.bfloat16)
sp_flash_model = apply_tensor_parallel(
    Llama(flash_model_config),
    sequence_parallel=True,
).to(device=device, dtype=torch.bfloat16)
sp_flash_model.load_state_dict(tp_flash_model.state_dict())

tp_flash_logits = tp_flash_model(input_ids)
sp_flash_logits = sp_flash_model(input_ids)
torch.testing.assert_close(sp_flash_logits, tp_flash_logits, rtol=2e-2, atol=2e-2)

torch.manual_seed(789)
logits_grad = torch.randn_like(tp_flash_logits)
tp_flash_logits.backward(logits_grad)
sp_flash_logits.backward(logits_grad)
for (tp_name, tp_parameter), (sp_name, sp_parameter) in zip(
    tp_flash_model.named_parameters(),
    sp_flash_model.named_parameters(),
):
    assert tp_name == sp_name
    torch.testing.assert_close(
        sp_parameter.grad,
        tp_parameter.grad,
        rtol=3e-2,
        atol=3e-2,
        msg=lambda message: f"Flash/bfloat16 gradient mismatch for {tp_name}: {message}",
    )

print(f"Rank {global_rank}: sequence-parallel tests passed")
dist.destroy_process_group()
