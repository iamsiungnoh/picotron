# Inspired by https://github.com/zhuzilin/ring-flash-attention
import os
import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func
from typing import Any, Optional, Tuple

import picotron.process_group_manager as pgm
# all_to_all added for implementing head
from picotron.context_parallel.cp_communications import ContextCommunicate, all_to_all
from picotron.utils import debug_test

def apply_context_parallel(model):
    os.environ["CONTEXT_PARALLEL"] = "1" if pgm.process_group_manager.cp_world_size > 1 else "0"
    return model

def ring_attention(q, k, v, sm_scale, is_causal,cp_zigzag_en=False):
    if cp_zigzag_en:
        assert is_causal == True, "zigzag ring is meaningless for causal=False"
        return ZigZagRingAttentionFunc.apply(q, k, v, sm_scale, is_causal)
    else:
        return RingAttentionFunc.apply(q, k, v, sm_scale, is_causal)



class ZigZagRingAttentionFunc(torch.autograd.Function):
    """
    Zigzag ring attention.

    Each CP rank's local q/k/v is assumed to be the concatenation of two zigzag
    chunks along the sequence dim: [low_chunk (index = cp_rank), high_chunk
    (index = 2*cp_world_size - 1 - cp_rank)], i.e. exactly the layout produced
    by `get_zigzag_indices`. This guarantees:
      - low_chunk positions are always < the sequence midpoint
      - high_chunk positions are always >= the sequence midpoint
    which is what makes the branching below correct.
    """

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, is_causal):
        comm = ContextCommunicate("comm")
        k_og = k.clone()
        v_og = v.clone()
        out, lse = None, None
        next_k, next_v = None, None

        block_seq_len = q.shape[2] // 2
        # "high" half of the local zigzag pair (the chunk mirrored from the back of the sequence)
        q1 = q[:, :, block_seq_len:, :]

        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k = comm.send_recv(k)
                next_v = comm.send_recv(v)
                comm.commit()

            if not is_causal:
                # Non-causal attention doesn't need load balancing: every rank always
                # attends to the full remote k/v with no masking.
                block_out, block_lse = ring_attention_forward(q, k, v, sm_scale, is_causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            elif step == 0:
                # Own chunk pair. A naive block-causal mask on the concatenated
                # [low|high] local sequence reproduces the true causal pattern.
                block_out, block_lse = ring_attention_forward(q, k, v, sm_scale, is_causal=True)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            elif step <= comm.rank:
                # Remote low chunk is fully in the past relative to *both* of our chunks
                # (attend with no mask); remote high chunk is fully in the future relative
                # to *both* of our chunks (contributes 0, so we skip computing it entirely).
                k0 = k[:, :, :block_seq_len, :]
                v0 = v[:, :, :block_seq_len, :]
                block_out, block_lse = ring_attention_forward(q, k0, v0, sm_scale, is_causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            else:
                # Remote chunks are fully in the past relative to our high chunk only;
                # our low chunk would get nothing from this remote pair, so only q1
                # (our high half) attends, and we scatter the result into that slice.
                block_out, block_lse = ring_attention_forward(q1, k, v, sm_scale, is_causal=False)
                out, lse = update_out_and_lse(
                    out, lse, block_out, block_lse,
                    slice_=(slice(None), slice(None), slice(block_seq_len, None), slice(None)),
                )

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)
        ctx.save_for_backward(q, k_og, v_og, out, lse.squeeze(-1))
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal

        kv_comm = ContextCommunicate("kv_comm")
        d_kv_comm = ContextCommunicate("d_kv_comm")

        block_seq_len = q.shape[2] // 2
        q1 = q[:, :, block_seq_len:, :]
        dout1 = dout[:, :, block_seq_len:, :]
        out1 = out[:, :, block_seq_len:, :]
        lse1 = softmax_lse[:, :, block_seq_len:]

        dq, dk, dv = None, None, None
        next_dk, next_dv = None, None
        next_k, next_v = None, None

        for step in range(kv_comm.world_size):
            if step + 1 != kv_comm.world_size:
                next_k = kv_comm.send_recv(k)
                next_v = kv_comm.send_recv(v)
                kv_comm.commit()

            # Mirror the exact same branching used in forward().
            half_k_v = False  # whether k_/v_ this step are only the low half of k,v
            high_q_only = False  # whether q_/dout_/out_/lse_ this step are only q1 (high half)

            if not is_causal:
                bwd_causal = False
                q_, k_, v_, dout_, out_, lse_ = q, k, v, dout, out, softmax_lse
            elif step == 0:
                bwd_causal = True
                q_, k_, v_, dout_, out_, lse_ = q, k, v, dout, out, softmax_lse
            elif step <= kv_comm.rank:
                bwd_causal = False
                q_, dout_, out_, lse_ = q, dout, out, softmax_lse
                k_, v_ = k[:, :, :block_seq_len, :], v[:, :, :block_seq_len, :]
                half_k_v = True
            else:
                bwd_causal = False
                q_, dout_, out_, lse_ = q1, dout1, out1, lse1
                k_, v_ = k, v
                high_q_only = True

            block_dq, block_dk, block_dv = ring_attention_backward(
                dout_, q_, k_, v_, out_, lse_, sm_scale, bwd_causal
            )

            # ---- accumulate dQ (into full-size buffer, in the right slice) ----
            if dq is None:
                dq = torch.zeros_like(q, dtype=torch.float32)
            if high_q_only:
                dq[:, :, block_seq_len:, :] += block_dq
            else:
                dq += block_dq

            # ---- pad block_dk/block_dv up to full k/v size before ring accumulation ----
            full_dk = torch.zeros_like(k, dtype=torch.float32)
            full_dv = torch.zeros_like(v, dtype=torch.float32)
            if half_k_v:
                full_dk[:, :, :block_seq_len, :] += block_dk
                full_dv[:, :, :block_seq_len, :] += block_dv
            else:
                full_dk += block_dk
                full_dv += block_dv

            if dk is None:
                dk = full_dk
                dv = full_dv
            else:
                d_kv_comm.wait()
                dk = full_dk + next_dk
                dv = full_dv + next_dv

            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k = next_k
                v = next_v

            next_dk = d_kv_comm.send_recv(dk)
            next_dv = d_kv_comm.send_recv(dv)
            d_kv_comm.commit()

        d_kv_comm.wait()

        dq = dq.to(q.dtype)
        next_dk = next_dk.to(k.dtype)
        next_dv = next_dv.to(v.dtype)

        return dq, next_dk, next_dv, None, None


class RingAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, is_causal):
        comm = ContextCommunicate("comm")
        #TODO(fmom): add flex attention
        #TODO(fmom): add flash attention
        #TODO(fmom): Find a better to save these tensors without cloning
        k_og = k.clone()
        v_og = v.clone()
        out, lse = None, None
        next_k, next_v = None, None

        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k = comm.send_recv(k)
                next_v = comm.send_recv(v)
                comm.commit()

            if not is_causal or step <= comm.rank:
                block_out, block_lse  = ring_attention_forward(
                    q, k, v, sm_scale, is_causal and step == 0
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                
            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)
        ctx.save_for_backward(q, k_og, v_og, out, lse.squeeze(-1))
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout, *args):

        q, k, v, out, softmax_lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal

        kv_comm = ContextCommunicate("kv_comm")
        d_kv_comm = ContextCommunicate("d_kv_comm")
        dq, dk, dv = None, None, None
        next_dk, next_dv = None, None
        
        block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        next_dk, next_dv = None, None
        next_k, next_v = None, None

        for step in range(kv_comm.world_size):
            if step + 1 != kv_comm.world_size:
                next_k = kv_comm.send_recv(k)
                next_v = kv_comm.send_recv(v)
                kv_comm.commit()

            if step <= kv_comm.rank or not is_causal:
                bwd_causal = is_causal and step == 0

                block_dq_buffer, block_dk_buffer, block_dv_buffer = ring_attention_backward(
                    dout, q, k, v, out, softmax_lse, sm_scale, bwd_causal
                )

                if dq is None:
                    dq = block_dq_buffer.to(torch.float32)
                    dk = block_dk_buffer.to(torch.float32)
                    dv = block_dv_buffer.to(torch.float32)
                else:
                    dq += block_dq_buffer
                    d_kv_comm.wait()
                    dk = block_dk_buffer + next_dk
                    dv = block_dv_buffer + next_dv
            elif step != 0:
                d_kv_comm.wait()
                dk = next_dk
                dv = next_dv

            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k = next_k
                v = next_v

            next_dk = d_kv_comm.send_recv(dk)
            next_dv = d_kv_comm.send_recv(dv)
            d_kv_comm.commit()

        d_kv_comm.wait()

        return dq, next_dk, next_dv, None, None

def ring_attention_forward(q, k, v, sm_scale, is_causal):
    batch_size, nheads, seqlen, d = q.shape
    S = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if is_causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, nheads, seqlen, seqlen)
        S.masked_fill_(causal_mask, float('-inf'))

    # Online softmax
    S_max = torch.max(S, dim=-1, keepdim=True)[0]
    exp_S = torch.exp(S - S_max)
    exp_sum = torch.sum(exp_S, dim=-1, keepdim=True)
    log_sum_exp = torch.log(exp_sum) + S_max
    P = exp_S / exp_sum
    O = torch.matmul(P, v)
    return O, log_sum_exp.squeeze(-1)

def ring_attention_backward(dO, Q, K, V, O, softmax_lse, sm_scale, is_causal):
    batch_size, nheads, seqlen, d = Q.shape
    
    # Recreate S and P from log_sum_exp
    S = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    if is_causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

    P = torch.exp(S - softmax_lse.unsqueeze(-1))
    # Step 1: Compute dV
    dV = torch.matmul(P.transpose(-2, -1), dO)
    # Step 2: Compute dP
    dP = torch.matmul(dO, V.transpose(-2, -1))
    # Step 3: Compute D
    D = torch.sum(dO * O, dim=-1, keepdim=True)
    # Step 4: Compute dS
    dS = P * (dP - D)
    # Apply causal mask to dS if is_causal is True
    if is_causal:
        dS = dS.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), 0)
    # Step 5: Compute dQ
    dQ = torch.matmul(dS, K) * sm_scale
    # Step 6: Compute dK
    dK = torch.matmul(dS.transpose(-2, -1), Q) * sm_scale
    return dQ, dK, dV

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_: Optional[Any] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    def _update(current_out, current_lse):
        # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
        # out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
        # For additional context and discussion, please refer to:
        # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
        current_out = current_out - F.sigmoid(block_lse - current_lse) * (current_out - block_out)
        current_lse = current_lse - F.logsigmoid(current_lse - block_lse)
        return current_out, current_lse
    
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.unsqueeze(dim=-1)

    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        return block_out, block_lse

    if slice_ is not None:
        out[slice_], lse[slice_] = _update(out[slice_], lse[slice_])
    else:
        out, lse = _update(out, lse)
        
    return out, lse

def update_rope_for_context_parallel(cos, sin):
    seq_len, _ = cos.size()
    cp_rank, cp_word_size = pgm.process_group_manager.cp_rank, pgm.process_group_manager.cp_world_size
    assert seq_len % cp_word_size == 0, f"Input sequence length ({seq_len}) must be divisible by cp_world_size ({cp_word_size})"
    size_per_partition = seq_len // cp_word_size
    start_idx, end_idx = cp_rank * size_per_partition, (cp_rank + 1) * size_per_partition
    return cos[start_idx:end_idx], sin[start_idx:end_idx]



def update_rope_for_context_parallel_zig_zag(cos, sin):
    seq_len, _ = cos.size()

    cp_rank = pgm.process_group_manager.cp_rank
    cp_world_size = pgm.process_group_manager.cp_world_size

    if cp_world_size == 1:
        return cos, sin

    indices = get_zigzag_indices(
        seq_len,
        cp_rank,
        cp_world_size,
        device=cos.device,
    )

    return (
        cos.index_select(0, indices),
        sin.index_select(0, indices),
    )

def get_zigzag_indices(seq_len, cp_rank, cp_world_size, device=None):
    assert seq_len % (2 * cp_world_size) == 0, (
        f"seq_len={seq_len} must be divisible by "
        f"2 * cp_world_size={2 * cp_world_size}"
    )

    chunk_size = seq_len // (2 * cp_world_size)

    chunk_ids = [
        cp_rank,
        2 * cp_world_size - cp_rank - 1,
    ]

    indices = []

    for chunk_id in chunk_ids:
        start = chunk_id * chunk_size
        end = start + chunk_size
        indices.append(
            torch.arange(
                start,
                end,
                device=device,
            )
        )
    indices = torch.cat(indices)
    debug_test(
        f"zigzag: seq_len={seq_len}, "
        f"chunk_size={chunk_size}, "
        f"chunk_ids={chunk_ids}, "
        f"indices={indices.tolist()}"
    )
    return indices



def sequence_to_head(x, group, cp_world_size):
    """Redistribute [B, H, S/CP, D] to [B, H/CP, S, D]."""
    batch_size, num_heads, local_seq_len, head_dim = x.shape
    if num_heads % cp_world_size != 0:
        raise ValueError(
            f"Number of heads ({num_heads}) must be divisible by the "
            f"context-parallel world size ({cp_world_size})"
        )

    local_num_heads = num_heads // cp_world_size

    #####################################################################################
    ###              TODO: Implement the sequence to head redistribution              ###
    ###       Input shape: [B, H, S/CP, D] (Batch, Heads, Sequence, Hidden_dim)       ###
    ###       Output shape: [B, H/CP, S, D] (Batch, Heads, Sequence, Hidden_dim)      ###
    #####################################################################################

    # Dimension 0 of the packed tensor identifies the destination rank.
    x = x.reshape(
        batch_size,
        cp_world_size,
        local_num_heads,
        local_seq_len,
        head_dim,
    )
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    x = all_to_all(x, group=group)

    # Dimension 0 now identifies the source rank. Preserve that rank order
    # while joining the received local sequence chunks.
    x = x.permute(1, 2, 0, 3, 4).contiguous()
    output =  x.reshape(
        batch_size,
        local_num_heads,
        cp_world_size * local_seq_len,
        head_dim,
    )

    ####################################################################################
    ###                            END of Implementation.                            ###
    ####################################################################################

    return output


def head_to_sequence(x, group, cp_world_size):
    """Redistribute [B, H/CP, S, D] to [B, H, S/CP, D]."""
    batch_size, local_num_heads, global_seq_len, head_dim = x.shape
    if global_seq_len % cp_world_size != 0:
        raise ValueError(
            f"Sequence length ({global_seq_len}) must be divisible by the "
            f"context-parallel world size ({cp_world_size})"
        )

    local_seq_len = global_seq_len // cp_world_size

    #####################################################################################
    ###              TODO: Implement the sequence to head redistribution              ###
    ###       Input shape: [B, H/CP, S, D] (Batch, Heads, Sequence, Hidden_dim)       ###
    ###       Output shape: [B, H, S/CP, D] (Batch, Heads, Sequence, Hidden_dim)      ###
    #####################################################################################

    # Send each sequence chunk back to the rank that originally owned it.
    x = x.reshape(
        batch_size,
        local_num_heads,
        cp_world_size,
        local_seq_len,
        head_dim,
    )
    x = x.permute(2, 0, 1, 3, 4).contiguous()
    x = all_to_all(x, group=group)

    # Received chunks are ordered by their source head rank. Join those chunks
    # to restore all heads for this rank's local sequence.
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    output =  x.reshape(
        batch_size,
        cp_world_size * local_num_heads,
        local_seq_len,
        head_dim,
    )

    ####################################################################################
    ###                            END of Implementation.                            ###
    ####################################################################################

    return output

def headwise_attention(q, k, v, is_causal):
    return HeadwiseContextParallel.apply(q, k, v, is_causal)


class HeadwiseContextParallel:
    """Apply Ulysses context parallelism directly around attention."""

    @staticmethod
    def apply(q, k, v, is_causal):

        group = pgm.process_group_manager.cp_group
        cp_world_size = pgm.process_group_manager.cp_world_size

        #######################################################################################
        ###  TODO: Implement the forward pass of headwise context parallelism               ###
        ###  Step 1: redistribute QKV                                                       ###
        ###  Step 2: compute attention with redistributed QKV                               ###
        ###         (flash_attn_func(q, k, v, causal=is_causal))                            ###
        ###  Step 3: redistribute the output back to the original shape                     ###
        #######################################################################################

        q = sequence_to_head(q, group, cp_world_size)
        k = sequence_to_head(k, group, cp_world_size)
        v = sequence_to_head(v, group, cp_world_size)

        # FlashAttention uses [batch, sequence, heads, head_dim].
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        out = flash_attn_func(
            q,
            k,
            v,
            causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous()
        output = head_to_sequence(out, group, cp_world_size)

        ####################################################################################
        ###                            END of Implementation.                            ###
        ####################################################################################

        return output