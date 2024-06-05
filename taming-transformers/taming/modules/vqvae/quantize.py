import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e     # 보통 1024
        self.e_dim = e_dim # 보통 256
        self.beta = beta   # 보통 0.25

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):   # z: [B, 256, 16, 16]
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()  # -> [B, 16, 16, 256]
        z_flattened = z.view(-1, self.e_dim)    # -> [B*16*16 ,256]
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())  # [B*16*16, 1024] -2 * matmul([B*16*16, 256], [256, 1024])

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # [B*16*16, 1]

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)  # min_encodings: [B*16*16, 1024]
        min_encodings.scatter_(1, min_encoding_indices, 1)  # one-hot vector만듦. 이 연산이 backprop이 안 됨.

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)  # matmul([B*16*16, 1024], [1024, 256]): [B*16*16, 256] -> view: [B, 16, 16, 256]
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
            torch.mean((z_q - z.detach()) ** 2)  # NOTE 버그 수정: 원코드에서는 self.beta = 0.25가 두번째 항에 곱해져 있었는데 논문의 식대로라면 첫번째 항에 곱해져 있어야 함.

        # preserve gradients
        z_q = z + (z_q - z).detach()   # backprop하면 z_q까지 온 gradient가 z로 점프해서 흘러감

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0) # -> [1024]
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # -> [B, 256, 16, 16]

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)  # [B, 256, 16, 16], tensor(scalar), ( tensor(scalar), [B*16*16, 1024], [B*16*16, 1] )

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
