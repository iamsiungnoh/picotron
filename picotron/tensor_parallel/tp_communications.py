import torch.distributed as dist
import torch
import picotron.process_group_manager as pgm
import torch.nn.functional as F

from typing import Tuple

def merge_first_two_dims(grad_output: torch.Tensor, input_: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge the first two dimensions of tensors."""
    return grad_output.contiguous().view(-1, *grad_output.shape[2:]), input_.contiguous().view(-1, *input_.shape[2:])

def split_tensor_along_last_dim(tensor, num_partitions):
    """Split a tensor along its last dimension into num_partitions chunks."""
    ###############################################################################
    # [Part 2] TODO: Split `tensor` into `num_partitions` equal chunks of size            #
    # `last_dim_size` along `last_dim`, and return the result.                   #
    ###############################################################################
    raise NotImplementedError
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
 

def gather_along_sequence_dim(x: torch.Tensor) -> torch.Tensor:
    """Gather equal sequence shards from the TP group along dimension 1."""
    tp_world_size = pgm.process_group_manager.tp_world_size
    if tp_world_size == 1:
        return x
    if x.dim() < 2:
        raise ValueError(f"Sequence-parallel tensors must have at least 2 dimensions, got shape {tuple(x.shape)}")

    # The *_into_tensor collectives concatenate/split along dimension 0, so
    # temporarily move the sequence dimension there.
    x_sequence_first = x.movedim(1, 0).contiguous()
    gathered_sequence_first = torch.empty(
        (x_sequence_first.size(0) * tp_world_size, *x_sequence_first.shape[1:]),
        dtype=x.dtype,
        device=x.device,
    )
    dist.all_gather_into_tensor(
        gathered_sequence_first,
        x_sequence_first,
        group=pgm.process_group_manager.tp_group,
    )
    return gathered_sequence_first.movedim(0, 1).contiguous()

def scatter_along_sequence_dim(x: torch.Tensor) -> torch.Tensor:
    """Return this TP rank's chunk of the sequence dimension."""
    tp_world_size = pgm.process_group_manager.tp_world_size
    if tp_world_size == 1:
        return x
    if x.dim() < 2:
        raise ValueError(f"Sequence-parallel tensors must have at least 2 dimensions, got shape {tuple(x.shape)}")
    if x.size(1) % tp_world_size != 0:
        raise ValueError(
            f"Sequence length ({x.size(1)}) must be divisible by TP size ({tp_world_size})"
        )

    return x.chunk(tp_world_size, dim=1)[pgm.process_group_manager.tp_rank].contiguous()

def reduce_scatter_along_sequence_dim(x: torch.Tensor) -> torch.Tensor:
    """Sum across the TP group and scatter the result along dimension 1."""
    tp_world_size = pgm.process_group_manager.tp_world_size
    if tp_world_size == 1:
        return x
    if x.dim() < 2:
        raise ValueError(f"Sequence-parallel tensors must have at least 2 dimensions, got shape {tuple(x.shape)}")
    if x.size(1) % tp_world_size != 0:
        raise ValueError(
            f"Sequence length ({x.size(1)}) must be divisible by TP size ({tp_world_size})"
        )

    x_sequence_first = x.movedim(1, 0).contiguous()
    local_sequence_length = x_sequence_first.size(0) // tp_world_size
    output_sequence_first = torch.empty(
        (local_sequence_length, *x_sequence_first.shape[1:]),
        dtype=x.dtype,
        device=x.device,
    )
    dist.reduce_scatter_tensor(
        output_sequence_first,
        x_sequence_first,
        group=pgm.process_group_manager.tp_group,
    )
    return output_sequence_first.movedim(0, 1).contiguous()

class GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather sequence shards in forward and reduce-scatter in backward."""

    @staticmethod
    def forward(ctx, x):
        return gather_along_sequence_dim(x)

    @staticmethod
    def backward(ctx, grad_output):
        return reduce_scatter_along_sequence_dim(grad_output)

class ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Scatter the sequence in forward and all-gather in backward."""

    @staticmethod
    def forward(ctx, x):
        return scatter_along_sequence_dim(x)

    @staticmethod
    def backward(ctx, grad_output):
        return gather_along_sequence_dim(grad_output)

class ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce-scatter the sequence in forward and all-gather in backward."""

    @staticmethod
    def forward(ctx, x):
        return reduce_scatter_along_sequence_dim(x)

    @staticmethod
    def backward(ctx, grad_output):
        return gather_along_sequence_dim(grad_output)

class CopyToModelParallelRegion(torch.autograd.Function):
    """
    Copy in forward pass, all-reduce in backward pass.
    This is the `f` function in the paper: https://arxiv.org/abs/1909.08053
    """
    @staticmethod
    def forward(ctx, x):
        ###############################################################################
        # [Part 2] TODO: Implement the forward pass                                            #
        ###############################################################################
        raise NotImplementedError
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
          return grad_output
        ###############################################################################
        # [Part 2] TODO: Implement the all_reduce                                              #
        # Hint: dist.all_reduce https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.all_reduce
        ###############################################################################
        raise NotImplementedError
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return grad_output

class ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    All-reduce in forward pass, identity in backward pass.
    This is the `g` function in the paper: https://arxiv.org/abs/1909.08053
    """
    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x
        ###############################################################################
        # # [Part 2] TODO: All-reduce `x` (sum) across the tensor-parallel group                 #
        # Hint: dist.all_reduce https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.all_reduce
        ###############################################################################
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        # raise NotImplementedError
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather in forward pass, split in backward pass."""
    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x
        last_dim = x.dim() - 1
        # Need contiguous tensors for collectives -> https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py#L321
        x = x.contiguous()
        # Step 1: allocate one empty buffer per TP rank, and place our own
        # shard into the slot matching our own rank (saves one copy — every
        # other rank's slot will be filled in by the collective below).
        tensor_list = [torch.empty_like(x) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = x
        ###############################################################################
        # [Part 2] TODO: Step 2: Gather `x` from every rank in the tensor-parallel group      #
        # Hint: https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.all_gather
        ###############################################################################
        raise NotImplementedError
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ###############################################################################
       
        ###############################################################################
        # [Part 2] Step 3 — concatenate the per-rank shards in `tensor_list` back into   #
        # the full-size tensor, along the dimension that was originally sharded       #
        # (`last_dim`). Don't forget to make the result contiguous.                   #
        # Hint: https://docs.pytorch.org/docs/2.13/generated/torch.cat.html           #
        ###############################################################################
        raise NotImplementedError
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # Split gradient according to TP size
        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()

class LinearWithAsyncAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        output = input_ @ weight.t() + bias if bias is not None else input_ @ weight.t()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        The key difference with "linear_with_all_reduce" is that the all reduce of input_ gradeint is before 
        the calculation of the gradient of weights and bias, instead of after. So we can overlap the computation and communication
        This is only applicable to Column Parallel Linear

        Before: grad_output -> grad_input, grad_weight, grad_bias  -> grad_input all reduce
        Now:    grad_output -> grad_input -> grad_input all reduce -> grad_weight, grad_bias
        """
        input_, weight = ctx.saved_tensors
        grad_input = grad_output @ weight # (b, s, out_size) @ (out_size, input_size) = (b, s, input_size)
        # all-reduce input gradient. 
        input_gradient_all_reduce_handle = dist.all_reduce(grad_input, group=pgm.process_group_manager.tp_group, async_op=True)
        # merge first two dims to allow matrix multiplication
        grad_output, input_ = merge_first_two_dims(grad_output, input_)     # grad_output, input_: (b, s, out_size), (b, s, input_size) -> (b*s, out_size), (b*s, input_size)
        grad_weight = grad_output.t() @ input_                              # (out_size, b*s) @ (b*s, input_size) -> (out_size, input_size)
        grad_bias = grad_output.sum(0) if ctx.use_bias else None
        input_gradient_all_reduce_handle.wait()
        return grad_input, grad_weight, grad_bias

def linear_with_all_reduce(x, weight, bias):
    input_parallel = CopyToModelParallelRegion.apply(x)
    output = F.linear(input_parallel, weight, bias) # XW_i^T + b, output is Y_i
    return output

def linear_with_async_all_reduce(x, weight, bias):
    return LinearWithAsyncAllReduce.apply(x, weight, bias)
