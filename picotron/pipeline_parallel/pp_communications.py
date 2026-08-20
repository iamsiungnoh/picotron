import os
import torch
import torch.distributed as dist
import picotron.process_group_manager as pgm

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"

def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    global STEP
    global VERBOSE

    ########################################################################################
    # [Part 2]                                                                             #
    # TODO: Prepare the point-to-point communication for the requested operation.          #
    # 1. Handle pipeline boundaries: the first stage cannot receive forward activations    #
    #    or send backward gradients, and the last stage cannot send forward activations    #
    #    or receive backward gradients. Return None for these no-op cases.                 #
    # 2. Based on the operation and its direction, select the neighboring pipeline rank    #
    #    and assign it to `src` for a receive or `dest` for a send. Forward communication  #
    #    moves toward the next stage; backward communication moves toward the previous     #
    #    stage.                                                                            #
    # 3. For a receive operation, assign `tensor` to a newly allocated empty tensor using  #
    #    `shapes`, `device`, and `dtype`, with `requires_grad=True`. For a send operation, #
    #    preserve the `tensor` supplied by the caller.                                     #
    # The common code below performs the asynchronous P2P operation and waits for it.      #
    ########################################################################################

    src = None
    dest = None

    if operation == 'recv_forward':
        raise NotImplementedError
    elif operation == 'send_forward':
        raise NotImplementedError
    elif operation == 'recv_backward':
        raise NotImplementedError
    elif operation == 'send_backward':
        raise NotImplementedError

    #######################################################################################
    #                                END of Implementation.                               #
    #######################################################################################

    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    if VERBOSE: print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} {pgm.process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}", flush=True)
    [req.wait() for req in dist.batch_isend_irecv([op])]
    torch.cuda.synchronize()
    if VERBOSE: STEP += 1
    return tensor if not is_send else None

def bidirectional_pipeline_communicate(operation, send_tensor, recv_shapes, device, dtype):
    global STEP
    global VERBOSE
    is_fwd = (operation == 'send_fwd_recv_bwd')
    if (is_fwd and pgm.process_group_manager.pp_is_last_stage) or (not is_fwd and pgm.process_group_manager.pp_is_first_stage): return None
    peer_rank = pgm.process_group_manager.pp_next_rank if is_fwd else pgm.process_group_manager.pp_prev_rank
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, send_tensor, peer_rank), dist.P2POp(dist.irecv, recv_tensor, peer_rank)])
    if VERBOSE: print(f"{operation} | sending {'next' if is_fwd else 'prev'} {pgm.process_group_manager.pp_rank} -> {peer_rank} | "f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> {pgm.process_group_manager.pp_rank} | "f"STEP {STEP=} | RANK:{pgm.process_group_manager.pp_rank}", flush=True)
    [req.wait() for req in reqs]
    torch.cuda.synchronize()
    if VERBOSE: STEP += 1
    return recv_tensor
