# This file tries to reproduce Algorithm 1 on the paper 
# A Simple Framework for Contrastive Learning of Visual Representations
# which is available at: https://arxiv.org/abs/2002.05709


import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR_Module(nn.Module):
    def __init__(self, f, g, batch_size, img_size, device):
        super(SimCLR_Module, self).__init__()
        self.f = f
        self.g = g
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device




    def forward(self, inputs):
        outputs = torch.stack(inputs).permute(1,0,4,2,3).reshape(self.batch_size,-1,self.img_size[0],self.img_size[1]).float()
        # print('from the pipe {}' .format(outputs.shape))
        outputs = self.f(outputs)
        # print('from resnet {}' .format(outputs.shape))
        outputs = self.g(outputs)
        # print('from regressor {}' .format(outputs.shape))

        return outputs



# compute l(i,j) in Algorithm 1
def _compute_l(i, j, s, temperature):
    assert s.shape[0] == s.shape[1]

    # compute the Sum in the denominator of the negative log
    # this is the exp of all the row in s
    auxiliary = torch.exp(s[i,:]/temperature)
    # here I exclude exp(s[i,i]/temperature)
    auxiliary = torch.cat([auxiliary[:i], auxiliary[i+1:]])
    # the I compute the sum
    Sum = torch.sum(auxiliary)

    # here I finally return the negative log of the quotient
    return -torch.log(torch.exp(s[i][j]/temperature) / Sum)



# compute pairwise similarity in Algorithm 1
def _compute_pairwise_similarity(z1, z2):
    assert z1.shape == z2.shape
    # N is the batch size
    N = z1.shape[0]
    # projection_d is the dimension of the projection
    projection_d = z1.shape[1]

    # composition of both projections with N vectors in one with 2*N vectors
    # alternating vectors from original projections between even and odd indices
    # in the new vector z
    z = torch.zeros(N*2, projection_d)
    for k in range(N):
            z[2*k] = z2[k]
            z[2*k+1] = z1[k]

    # prepare the cosine similarity
    cos = nn.CosineSimilarity(dim=1)

    # Let's suppose 2*N = 4
    # and projection_d = 5, then 

    # if     |1 2 3 4 5|
    #    z = |4 5 6 7 8|
    #        |7 8 9 1 2|
    #        |1 8 9 1 2|

    # therefore z.repeat(2*N,1) will be:
    #        |1 2 3 4 5|
    #        |4 5 6 7 8|
    #        |7 8 9 1 2|
    #    z = |1 8 9 1 2|
    #        |1 2 3 4 5|
    #        |4 5 6 7 8|
    #        |7 8 9 1 2|
    #        |1 8 9 1 2|
    #        |1 2 3 4 5|
    #        |4 5 6 7 8|
    #        |7 8 9 1 2|
    #        |1 8 9 1 2|
    #        |1 2 3 4 5|
    #        |4 5 6 7 8|
    #        |7 8 9 1 2|
    #        |1 8 9 1 2|

    # and z.repeat(1,2*N) will be:
    #        |1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5|
    #    z = |4 5 6 7 8 4 5 6 7 8 4 5 6 7 8 4 5 6 7 8|
    #        |7 8 9 1 2 7 8 9 1 2 7 8 9 1 2 7 8 9 1 2|
    #        |1 8 9 1 2 1 8 9 1 2 1 8 9 1 2 1 8 9 1 2|

    # but finally z.repeat(1,2*N).reshape(2*N*(2*N),-1)) will be:
    #        |1 2 3 4 5|
    #        |1 2 3 4 5|
    #        |1 2 3 4 5|
    #    z = |1 2 3 4 5|
    #        |4 5 6 7 8|
    #        |4 5 6 7 8|
    #        |4 5 6 7 8|
    #        |4 5 6 7 8|
    #        |7 8 9 1 2|
    #        |7 8 9 1 2|
    #        |7 8 9 1 2|
    #        |7 8 9 1 2|
    #        |1 8 9 1 2|
    #        |1 8 9 1 2|
    #        |1 8 9 1 2|
    #        |1 8 9 1 2|

    # hence, computing the similarity between this two vector arranges
    # will produce the similarity among all possible combinations of vectors
    # in z
    s = cos(z.repeat(2*N,1), z.repeat(1,2*N).reshape(2*N*(2*N),-1))
    return s.reshape(2*N,2*N)





# compute the loss (Algorithm 1) between two projections
def compute_loss(z1, z2, temperature):
    # N is the batch size
    N = z1.shape[0]

    # first, I compute pairwise similarity
    s = _compute_pairwise_similarity(z1, z2)

    # finally I compute the loss
    Sum = 0.0
    for k in range(N):
        Sum += (_compute_l(2*k+1,2*k, s, temperature) + _compute_l(2*k, 2*k+1, s, temperature))

    return Sum / 2*N
