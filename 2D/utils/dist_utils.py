import torch
import math
import os
import re
import random
import numpy as np
import torch.distributed as dist


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def init_distributed_mode(args):

    try:
        # get env info
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])
        print('rank:{}, world_size:{}, local_rank:{}, node_list:{}'.format(args.rank, args.world_size, args.local_rank, node_list))
        # set ip address
        node_parts = re.findall('[0-9]+', node_list)
        host_ip = '{}.{}.{}.{}'.format(node_parts[1],node_parts[2],node_parts[3],node_parts[4])
        # port should not be used
        port = args.port
        # initialize tcp method
        init_method = 'tcp://{}:{}'.format(host_ip, port)  
        # initialize progress
        dist.init_process_group("nccl", init_method=init_method, world_size=args.world_size, rank=args.rank)
        # set device for each node
        torch.cuda.set_device(args.local_rank)
        print('ip: {}'.format(host_ip))

    except:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = torch.distributed.get_rank()
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):

        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
