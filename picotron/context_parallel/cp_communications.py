import os
import torch
from torch import distributed as dist
from typing import List

import picotron.process_group_manager as pgm

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"

####################################################################################
###                 Given AlltoALL implementation, do not modify                 ###
####################################################################################

def _all_to_all(input_, group):
    """Perform an equal-split all-to-all exchange along dimension 0."""
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input_

    if input_.size(0) % world_size != 0:
        raise ValueError(
            f"Dimension 0 with size {input_.size(0)} must be divisible "
            f"by the context-parallel world size ({world_size})"
        )

    send_buffer = torch.empty(
        input_.shape, dtype=input_.dtype, device=input_.device
    )
    send_buffer.copy_(input_)
    recv_buffer = torch.empty(
        input_.shape, dtype=input_.dtype, device=input_.device
    )
    dist.all_to_all_single(recv_buffer, send_buffer, group=group)
    return recv_buffer


class AllToAll(torch.autograd.Function):
    """Autograd-aware equal-split all-to-all collective."""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _all_to_all(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_to_all(grad_output, ctx.group), None


def all_to_all(input_, group=None):
    """
    Exchange equal-sized dimension-0 chunks across context-parallel ranks.

    The caller is responsible for packing the input so dimension 0 contains
    one chunk for each destination rank. Received chunks are ordered by source
    rank along dimension 0. Backward performs the same exchange in reverse.
    """
    if group is None:
        group = pgm.process_group_manager.cp_group
    return AllToAll.apply(input_, group)

####################################################################################
###                     END of Given AlltoALL implementation.                    ###
####################################################################################

class ContextCommunicate:
    def __init__(self, msg: str = ""):
        global STEP
        global VERBOSE
        self._pending_operations: List[dist.P2POp] = []
        self._active_requests = None
        self.rank = pgm.process_group_manager.cp_rank
        self.world_size = pgm.process_group_manager.cp_world_size
        self.send_rank = pgm.process_group_manager.cp_send_rank
        self.recv_rank = pgm.process_group_manager.cp_recv_rank
        if VERBOSE: print(f"RingComm ({msg}) | initialized | RANK:{self.rank} | "f"WORLD_SIZE:{self.world_size} | SEND_RANK:{self.send_rank} | "f"RECV_RANK:{self.recv_rank}", flush=True)

    def send_recv(self, tensor_to_send, recv_tensor=None):
        if recv_tensor is None:
            result_tensor = torch.zeros_like(tensor_to_send)
        else:
            result_tensor = recv_tensor

        ###############################################################################
        # [Part 2] TODO: Build the two P2POp descriptors for this ring step:                   #
        #   - one that will isend `tensor_to_send` to `self.send_rank`                #
        #   - one that will irecv into `result_tensor` from `self.recv_rank`          #
        # Both must use group=pgm.process_group_manager.cp_group.                     #
        # Hint: https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.P2POp
        ###############################################################################
        raise NotImplementedError
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

        
        self._pending_operations.extend([send_operation, recv_operation])

        if VERBOSE:
            print(f"RingComm | send_recv | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:sending | TO:{self.send_rank} | TENSOR:{tensor_to_send}", flush=True)
            print(f"RingComm | send_recv | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:receiving | FROM:{self.recv_rank} | TENSOR:{result_tensor}", flush=True)
        return result_tensor

    def commit(self):
        if self._active_requests is not None: raise RuntimeError("Commit called twice")
        ###############################################################################
        # [Part 2] TODO: Launch all queued operations in `self._pending_operations` as a       #
        # single batched P2P call, and store the returned request handles in         #
        # `self._active_requests`.     #
        # Hint: https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.batch_isend_irecv
        ###############################################################################

        raise NotImplementedError
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        if VERBOSE: print(f"RingComm | commit | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:committed | NUM_OPS:{len(self._pending_operations) // 2}", flush=True)

    def wait(self):
        if self._active_requests is None: raise RuntimeError("Wait called before commit")
        for i, request in enumerate(self._active_requests):
            request.wait()
            if VERBOSE:
                operation_type = "send" if i % 2 == 0 else "receive"
                peer_rank = self.send_rank if operation_type == "send" else self.recv_rank
                print(f"RingComm | wait | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:completed_{operation_type} | "f"{'FROM' if operation_type == 'receive' else 'TO'}:{peer_rank}", flush=True)
        torch.cuda.synchronize()
        self._active_requests = None
        self._pending_operations = []
        if VERBOSE: print(f"RingComm | wait | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:all_operations_completed", flush=True)
