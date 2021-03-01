import torch
import torch.nn.functional as f
import torch.distributed as dist


LARGE_NUM = 1e9










def contrastive_loss(hidden1, hidden2,
                     hidden_norm=True,
                     temperature=1.0,
                     local_rank=0,
                     world_size=1,
                     device='cpu'):
        """Compute loss for model.

        Args:
         hidden1: hidden vector (`Tensor`) of shape (bsz, dim).
         hidden2: hidden vector (`Tensor`) of shape (bsz, dim).
         hidden_norm: whether or not to use normalization on hidden vector.
         temperature: a `floating` number for temperature scaling.
         local_rank: the process identifier
         world_size: if it is 1, we do not have to use inter-process communication among different GPUs
         weights: a weighting number of vector.
        
        Returns:
         A loss scalar.
         The logits for contrastive prediction task.
         The labels for contrastive prediction task.
        """

        # Get (normalized) hidden1 and hidden2.
        if hidden_norm:
                hidden1 = f.normalize(hidden1, dim=1, p=2)
                hidden2 = f.normalize(hidden2, dim=1, p=2)
        
        assert(hidden1.shape == hidden2.shape)
        batch_size = hidden1.shape[0]
        dimensionality = hidden1.shape[1]


        # Gather hidden1/hidden2 across replicas and create local labels.
        if world_size > 1:
                hidden1_large = _cross_replica_concat(hidden1, world_size, batch_size, dimensionality, device)
                hidden2_large = _cross_replica_concat(hidden2, world_size, batch_size, dimensionality, device)
                enlarged_batch_size = hidden1_large.shape[0]
                labels_idx = torch.tensor(range(batch_size)) + local_rank * batch_size

                labels = f.one_hot(labels_idx, enlarged_batch_size * 2).to(device)
                masks = f.one_hot(labels_idx, enlarged_batch_size).to(device)
        else:
                hidden1_large = hidden1
                hidden2_large = hidden2
                labels_idx = torch.tensor(range(batch_size))

                labels = f.one_hot(labels_idx, batch_size * 2).to(device)
                masks = f.one_hot(labels_idx, batch_size).to(device)
 
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM

        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM

        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

        loss_a = _softmax_cross_entropy(labels, torch.cat([logits_ab, logits_aa], 1))
        loss_b = _softmax_cross_entropy(labels, torch.cat([logits_ba, logits_bb], 1))

        loss = loss_a + loss_b

        return loss, logits_ab, labels




















def _cross_replica_concat(tensor, world_size, minibatch_size, dimensionality, device):
        """Reduce a concatenation of the `tensor` across GPU cores.

        Args:
         tensor: tensor to concatenate

        Returns:
         Tensor of the same rank as `tensor` with first dimension `num_replicas`
         times larger.
        """
        tensor_list = [torch.zeros(minibatch_size, dimensionality).to(device) for i in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, 0)





# define "soft" cross-entropy with pytorch tensor operations
# this function is a copy-paste from:
# https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501/2 
def _softmax_cross_entropy(targets, inputs):
        logprobs = f.log_softmax (inputs, dim = 1)
        return  -(targets * logprobs).sum() / inputs.shape[0]

