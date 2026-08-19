import math
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import picotron.process_group_manager as pgm
from picotron.tensor_parallel.tp_communications import (
    CopyToModelParallelRegion,
    GatherFromModelParallelRegion,
    GatherFromSequenceParallelRegion,
    ReduceFromModelParallelRegion,
    ReduceScatterToSequenceParallelRegion,
    linear_with_all_reduce,
    linear_with_async_all_reduce,
)
from picotron.utils import debug_test

def apply_tensor_parallel(model, sequence_parallel=False):

    model.sequence_parallel = sequence_parallel

    def _replace_fused_qkv(attention_module):
        """
        NEW: replace attention.qkv_proj (plain nn.Linear from FuseQKVAttention)
        with a TP-aware FusedQKVColumnParallelLinear, and update the attention
        module's cached _qkv_split_sizes to the SHARDED (per-rank) sizes so
        forward()'s torch.split works on the sharded output, not the original
        full-size split computed in FuseQKVAttention.__init__.
        """
        linear_layer = attention_module.qkv_proj
        new_linear_layer = FusedQKVColumnParallelLinear(
            in_features=linear_layer.in_features,
            num_heads=attention_module.num_heads,
            num_key_value_heads=attention_module.num_key_values,
            head_dim=attention_module.head_dim,
            bias=linear_layer.bias is not None,
        )
        attention_module.qkv_proj = new_linear_layer
        # CHANGED: was the full (q_out, kv_out, kv_out) computed pre-TP;
        # now must reflect this rank's local shard sizes.
        attention_module._qkv_split_sizes = new_linear_layer.qkv_split_sizes_per_partition

    def _replace_module(_module, _linear_proj_name, _style, args={},vocab_padding_en=False):
        assert _style in ["column", "row", 'vocab']
        linear_layer = getattr(_module, _linear_proj_name)
        
        if _style == "column":
            if vocab_padding_en and  _linear_proj_name == "final_proj":
                new_linear_layer = VocabPadFinalOutput(
                    in_features=linear_layer.in_features,
                    vocab_size=linear_layer.out_features,
                    bias=linear_layer.bias is not None,
                    gather_output=args.get(
                        "gather_output",
                        False,
                    ),
                )
            else:
                new_linear_layer = ColumnParallelLinear(
                    in_features=linear_layer.in_features,
                    out_features=linear_layer.out_features,
                    bias=linear_layer.bias is not None,
                    gather_output=args.get("gather_output", False)
                    sequence_parallel=sequence_parallel,
                )

        elif _style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                sequence_parallel=sequence_parallel,
            )
        else:
            if vocab_padding_en:
                new_linear_layer = VocabParallelEmbeddingPadding(
                    num_embeddings=linear_layer.num_embeddings,
                    embedding_dim=linear_layer.embedding_dim,
                )
            else:
                new_linear_layer = VocabParallelEmbedding(
                                num_embeddings=linear_layer.num_embeddings,
                                embedding_dim=linear_layer.embedding_dim,
                            )

        setattr(_module, _linear_proj_name, new_linear_layer)

    module_linear_name_stype_mapping_list = [
        ("attention", "q_proj", "column"),
        ("attention", "k_proj", "column"),
        ("attention", "v_proj", "column"),
        ("attention", "out_proj", "row"),
        ("mlp", "up_proj", "column"),
        ("mlp", "gate_proj", "column"),
        ("mlp", "down_proj", "row"),
    ]
    

    for layer in model.decoder_layers:
        layer.input_layernorm.sequence_parallel = sequence_parallel
        layer.post_attention_layernorm.sequence_parallel = sequence_parallel
        if model.model_config.fuse_qkv_en:
            _replace_fused_qkv(layer.attention)
            _replace_module(layer.attention, "out_proj", "row")
            _replace_module(layer.mlp, "up_proj", "column")
            _replace_module(layer.mlp, "gate_proj", "column")
            _replace_module(layer.mlp, "down_proj", "row")
       
            
        else:
            for module_name, linear_proj_name, style in module_linear_name_stype_mapping_list:
                _replace_module(getattr(layer, module_name), linear_proj_name, style)
    vocab_padding_en = model.model_config.vocab_padding_en
    _replace_module(
        model,
        "embedding",
        "vocab",
        vocab_padding_en=vocab_padding_en,
    )
    model.final_norm.sequence_parallel = sequence_parallel
    _replace_module(
        model,
        "final_proj",
        "column",
        args={"gather_output": True},
        vocab_padding_en=vocab_padding_en,
    )
    
    return model

class ColumnParallelLinear(torch.nn.Module):
    """Column Parallel Linear layer
    Y = XW + b, where weight matrix W is parallelized along its second dimension. W = [W_1, ..., W_p]
    This module returns the results of Y_i = XW_i + b_i in the forward method, Y_i is parallelized in the second dimension.
    Arguments:
        in_features: first dimension of weight matrix W.
        out_features: second dimension of weight matrix W.
        bias: If true, add bias
        init_method: method to initialize weights
        gather_output: If true, gather the output from all the partitions. This is used for the last linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        async_all_reduce: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        assert out_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output
        self.async_all_reduce = async_all_reduce
        self.sequence_parallel = sequence_parallel
        if self.sequence_parallel and self.async_all_reduce:
            raise ValueError("Sequence parallelism cannot be combined with async input-gradient all-reduce")
        # Allocate space for the weight and bias
        # Note: torch.nn.functional.linear performs XW^T + b so we exchange the order of dimensions
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features)) # W_i
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with the default initialization method used for nn.Linear in PyTorch
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.output_size_per_partition
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        if self.sequence_parallel:
            input_parallel = GatherFromSequenceParallelRegion.apply(x)
            output = F.linear(input_parallel, self.weight, self.bias)
        elif self.async_all_reduce:
            output = linear_with_async_all_reduce(x, self.weight, self.bias) 
        else:
            output = linear_with_all_reduce(x, self.weight, self.bias) 
        if self.gather_output:
            output = GatherFromModelParallelRegion.apply(output)
        return output
    
class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    Y = XW + b. W is parallelized along its first dimension and X along its second dimension as:
               -   -
              | W_1 |
              | .   |
          W = | .   |        X = [X_1, ..., X_p]
              | .   |
              | W_p |
               -   -
    We assume that X is already parallelized. This is the case after ColumnParallelLinear.
    This module returns the results of Y = sum(X_i * W_i + b_i) in the forward method.
    Arguments:
        in_features: first dimension of matrix W.
        out_features: second dimension of matrix W.
        bias: If true, add bias
        init_method: method to initialize weights.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool, sequence_parallel: bool = False):
        super(RowParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        self.sequence_parallel = sequence_parallel
        assert in_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with same dtype and device as self.weight
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)    
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.input_size_per_partition
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        # X_i * W_i^T + b
        output_parallel = F.linear(x, self.weight)
        if self.sequence_parallel:
            output = ReduceScatterToSequenceParallelRegion.apply(output_parallel)
            # The bias is replicated across TP ranks, while each rank sees only
            # a sequence shard. Sum its gradient across TP in backward.
            bias = None if self.bias is None else CopyToModelParallelRegion.apply(self.bias)
        else:
            # All-reduce across all the partitions.
            output = ReduceFromModelParallelRegion.apply(output_parallel)
            bias = self.bias
        return output if bias is None else output + bias
    
class VocabParallelEmbeddingPadding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Optional[int] = None,
            max_norm: Optional[float] = None,
            norm_type: float = 2.0,
            scale_grad_by_freq: bool = False,
            sparse: bool = False
        ):
            super(VocabParallelEmbeddingPadding, self).__init__()
    
            self.tp_world_size = pgm.process_group_manager.tp_world_size
            self.tp_rank = pgm.process_group_manager.tp_rank
            # Original vocabulary size.
            self.num_embeddings = num_embeddings # original vocab size
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse

            # Pad vocabulary size so that it is divisible by TP world size.
            self.padded_num_embeddings = (math.ceil(self.num_embeddings / self.tp_world_size)* self.tp_world_size)
            

            # Divide the weight matrix along the vocaburaly dimension.
            self.vocab_start_index, self.vocab_end_index = self._vocab_range_from_global_vocab_size(
                self.num_embeddings, pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
            )
            self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
    
            self.weight = nn.Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))
            debug_test(
                f"[Vocab][rank {self.tp_rank}] "
                f"original_vocab={self.num_embeddings}, "
                f"padded_vocab={self.padded_num_embeddings}, "
                f"world_size={self.tp_world_size}, "
                f"local_range=[{self.vocab_start_index}, {self.vocab_end_index}), "
                f"local_vocab_size={self.num_embeddings_per_partition}, "
                f"padding_rows={max(0, self.vocab_end_index - self.num_embeddings)}"
            )
            self.reset_parameters()
        
    def _vocab_range_from_global_vocab_size(self, global_vocab_size: int, rank: int, world_size: int):
         
        padded_vocab_size = (
            math.ceil(global_vocab_size / world_size) * world_size
        )
        per_partition_vocab_size = padded_vocab_size // world_size
        # vocab_range_from_per_partition_vocab_size
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    def reset_parameters(self):
        """
        Initialize a padded global embedding table and then select
        the partition belonging to this TP rank.
        """
        master_weight = torch.empty(
            self.padded_num_embeddings, 
            self.embedding_dim, 
            dtype=self.weight.dtype,
            device=self.weight.device, 
            requires_grad=False
        )
        torch.nn.init.normal_(master_weight, mean=0.0, std=1.0)
        # Split the model into size of self.num_embeddings_per_partition
        weight_list = torch.split(master_weight, self.num_embeddings_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()
        debug_test(
            f"[Vocab][rank {self.tp_rank}] "
            f"embedding_weight_shape={tuple(self.weight.shape)}"
        )
    def forward(self, x):
        """
        Performs an embedding lookup for input tokens in the parallelized embedding layer
        1. Masks tokens that fall outside the specified vocabulary range and adjusts the input
        2. Performs embedding lookups for valid tokens, setting embeddings of out-of-vocabulary tokens to zero
        3. Reduces the embeddings across model parallel GPUs using all-reduce for synchronization
        """
        # Build the mask for out-of-vocabulary tokens.
        input_mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)
        # padded token IDs >= original num_embeddings are not real tokens.
        invalid_token_mask = x >= self.num_embeddings
        input_mask = input_mask | invalid_token_mask
        # Mask the input.
        masked_input = x.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        # Get the embeddings for the valid tokens.
        local_padding_idx = None
        if self.padding_idx is not None:
            if self.vocab_start_index <= self.padding_idx < self.vocab_end_index:
                local_padding_idx = (
                    self.padding_idx - self.vocab_start_index
                )
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            local_padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Embedding of out-of-vocabulary tokens is set to 0.
        output_parallel[input_mask, :] = 0.0
        output = ReduceFromModelParallelRegion.apply(output_parallel)
        return output
    

class VocabPadFinalOutput(nn.Module):
    def __init__(
        self,
        in_features: int,
        vocab_size: int,
        bias: bool = False,
        gather_output: bool = True,
    ):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features

        # Real/original vocabulary size.
        self.vocab_size = vocab_size

        # Pad vocabulary so it is divisible by TP world size.
        self.padded_vocab_size = (
            (vocab_size + self.tp_world_size - 1)
            // self.tp_world_size
            * self.tp_world_size
        )

        self.vocab_size_per_partition = (
            self.padded_vocab_size // self.tp_world_size
        )

        self.vocab_start_index = (
            self.tp_rank * self.vocab_size_per_partition
        )
        self.vocab_end_index = (
            self.vocab_start_index
            + self.vocab_size_per_partition
        )

        self.gather_output = gather_output

        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_partition,
                self.in_features,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.vocab_size_per_partition)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize a padded global lm_head and select this TP rank's shard.
        """
        master_weight = torch.empty(
            self.padded_vocab_size,
            self.in_features,
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False,
        )

        # Same style as nn.Linear / your ColumnParallelLinear.
        k = 1 / self.in_features
        bound = math.sqrt(k)

        torch.nn.init.uniform_(
            master_weight,
            -bound,
            bound,
        )

        weight_list = torch.split(
            master_weight,
            self.vocab_size_per_partition,
            dim=0,
        )

        self.weight.data.copy_(
            weight_list[self.tp_rank].contiguous()
        )

    def forward(self, x):
        """
        x: [..., hidden_size]

        Each rank computes logits for its local vocab partition.
        If gather_output=True, gather all vocab partitions and
        remove padded logits.
        """

        # Local vocab logits.
        output_parallel = F.linear(
            x,
            self.weight,
            self.bias,
        )

        if not self.gather_output:
            return output_parallel

        # Gather vocab shards:
        # [..., vocab_per_rank] -> [..., padded_vocab_size]
        output = GatherFromModelParallelRegion.apply(
            output_parallel
        )

        # Remove fake padded vocabulary entries.
        output = output[..., :self.vocab_size]
        debug_test(
            f"[VocabOutput][rank {self.tp_rank}] "
            f"original_vocab={self.vocab_size}, "
            f"padded_vocab={self.padded_vocab_size}, "
            f"local_vocab_size={self.vocab_size_per_partition}, "
            f"local_range=[{self.vocab_start_index}, {self.vocab_end_index})"
        )
        return output

class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        super(VocabParallelEmbedding, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        self.weight = nn.Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))

        self.reset_parameters()
    
    def _vocab_range_from_global_vocab_size(self, global_vocab_size: int, rank: int, world_size: int):
        #TODO: do some padding for the vocab size
        assert global_vocab_size % world_size == 0, f"{global_vocab_size} is not divisible by {world_size}"
        per_partition_vocab_size = global_vocab_size // world_size
        # vocab_range_from_per_partition_vocab_size
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    def reset_parameters(self):
        master_weight = torch.empty(
            self.num_embeddings, 
            self.embedding_dim, 
            dtype=self.weight.dtype,
            device=self.weight.device, 
            requires_grad=False
        )
        torch.nn.init.normal_(master_weight, mean=0.0, std=1.0)
        # Split the model into size of self.num_embeddings_per_partition
        weight_list = torch.split(master_weight, self.num_embeddings_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        """
        Performs an embedding lookup for input tokens in the parallelized embedding layer
        1. Masks tokens that fall outside the specified vocabulary range and adjusts the input
        2. Performs embedding lookups for valid tokens, setting embeddings of out-of-vocabulary tokens to zero
        3. Reduces the embeddings across model parallel GPUs using all-reduce for synchronization
        """
        # Build the mask for out-of-vocabulary tokens.
        input_mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)
        # Mask the input.
        masked_input = x.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        # Get the embeddings for the valid tokens.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Embedding of out-of-vocabulary tokens is set to 0.
        output_parallel[input_mask, :] = 0.0
        output = ReduceFromModelParallelRegion.apply(output_parallel)
        return output

class FusedQKVColumnParallelLinear(nn.Module):
    """
    TP-aware column-parallel linear for FuseQKVAttention's fused qkv_proj.

    Unlike a plain ColumnParallelLinear (which slices its output dim into
    tp_world_size EQUAL CONTIGUOUS chunks), the fused output here is
    [Q | K | V] and each rank must receive a WHOLE number of heads from
    Q, K, AND V simultaneously -- otherwise the per-head reshape/repeat_interleave
    logic in FuseQKVAttention.forward() silently computes wrong results.

    Strategy: split Q into tp_world_size whole-head chunks, split K and V
    likewise (using num_key_value_heads), then concatenate THIS RANK's
    Q-chunk + K-chunk + V-chunk into the local weight.
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        bias: bool = False,
        async_all_reduce: bool = False,
    ) -> None:
        super().__init__()
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        assert num_heads % self.tp_world_size == 0, \
            "num_attention_heads should be divisible by tp world size"
        assert num_key_value_heads % self.tp_world_size == 0, \
            "num_key_value_heads should be divisible by tp world size"

        self.in_features = in_features
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        self.heads_per_partition = num_heads // self.tp_world_size
        self.kv_heads_per_partition = num_key_value_heads // self.tp_world_size

        self.q_size_per_partition = self.heads_per_partition * head_dim
        self.kv_size_per_partition = self.kv_heads_per_partition * head_dim
        self.output_size_per_partition = self.q_size_per_partition + 2 * self.kv_size_per_partition
        self.out_features = num_heads * head_dim + 2 * num_key_value_heads * head_dim

        # NEW: per-rank split sizes, consumed by FuseQKVAttention.forward() via
        # attention_module._qkv_split_sizes (set by apply_tensor_parallel after replacement)
        self.qkv_split_sizes_per_partition = (
            self.q_size_per_partition,
            self.kv_size_per_partition,
            self.kv_size_per_partition,
        )

        self.async_all_reduce = async_all_reduce
        self.gather_output = False  # QKV output stays local per rank; used directly for local attention heads

        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim

        master_q = torch.empty(q_out, self.in_features, dtype=self.weight.dtype, device=self.weight.device)
        master_k = torch.empty(kv_out, self.in_features, dtype=self.weight.dtype, device=self.weight.device)
        master_v = torch.empty(kv_out, self.in_features, dtype=self.weight.dtype, device=self.weight.device)

        bound = math.sqrt(1 / self.in_features)
        torch.nn.init.uniform_(master_q, -bound, bound)
        torch.nn.init.uniform_(master_k, -bound, bound)
        torch.nn.init.uniform_(master_v, -bound, bound)

        # Split each of Q, K, V independently into tp_world_size whole-head
        # chunks, THEN concatenate this rank's Q/K/V chunks together.
        q_chunks = torch.split(master_q, self.q_size_per_partition, dim=0)
        k_chunks = torch.split(master_k, self.kv_size_per_partition, dim=0)
        v_chunks = torch.split(master_v, self.kv_size_per_partition, dim=0)

        local_weight = torch.cat(
            [q_chunks[self.tp_rank], k_chunks[self.tp_rank], v_chunks[self.tp_rank]],
            dim=0,
        )
        self.weight.data = local_weight.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.async_all_reduce:
            output = linear_with_async_all_reduce(x, self.weight, self.bias)
        else:
            output = linear_with_all_reduce(x, self.weight, self.bias)
        return output
