from functools import partial
from itertools import islice, cycle

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from rotary_embedding_torch import RotaryEmbedding, broadcat
from axial_positional_embedding import AxialPositionalEmbedding
from g_mlp_pytorch import gMLPBlock
from torch.cuda.amp import autocast

from transformer_dalle.reversible import ReversibleSequence, SequentialSequence
from transformer_dalle.attention import * 
from transformer_dalle.transformer import *
from performer_pytorch import *
import torchvision

## helpers

def eval_decorator(fn):
    def inner(model, *args, **kwargs):  # args: (text[:1]), kwargs: {'filter_thres': 0.9}
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs) # generated image: [1, 3, 256, 256]
        model.train(was_training)  # model.training을 was_training값으로 바꿈
        return out
    return inner

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x.permute(0,3,1,2))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class TransformerLM_i2t(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens,
        num_img_tokens,
        max_seq_len,
        max_img_num,
        padding_idx,
        condition_len,
        reversible = False,
        causal = True,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        emb_dropout=0.,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_types = None,
        image_fmap_size = None,
        sparse_attn = False,
        stable = False,
        shift_tokens = False,
        rotary_emb = False,
        tie_embed=False,

        nb_features = None,
        generalized_attention = False,
        kernel_fn=nn.ReLU(),
        no_projection = False,
        profile = True
    ):
        super().__init__()
        
        self.dim=dim
        self.num_tokens = num_tokens   #= text vocab size
        self.max_seq_len = max_seq_len
        self.max_img_num = max_img_num
        self.condition_len = condition_len
        self.emb_dropout = emb_dropout
        self.rotary_emb = rotary_emb
        
        ##! Emb and Enc for each image and report
        # IMAGE embedding and positioncal encoding
        # self.image_token_emb = nn.Embedding(num_img_tokens, dim, padding_idx)
        self.img_embeddings = nn.Linear(2048, dim)
        
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(image_fmap_size, image_fmap_size))
        self.cnn = CNNmodel()

        # REPORT embedding and positioncal encoding
        # num_tokens = text vocab size
        self.token_emb = nn.Embedding(num_tokens, dim, padding_idx)
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.layer_pos_emb = Always(None)
        
        self.dropout = nn.Dropout(emb_dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)


        ##! Get attention function and Set layers and Ready
        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        for ind, sparse_attn, attn_type in zip(range(depth), sparse_layer, attn_type_layer):
            if attn_type == 'conditioned_i2t':
                attn_class = partial(ConditionedCausalAttention_i2t, 
                                     seq_len = max_seq_len, condition_len=condition_len, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'favor+':
                attn_class = FastAttention(dim_head, nb_features, causal = causal, condition_len = condition_len, 
                                           generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection, profile=profile)
            elif attn_type == 'full':
                attn_class = partial(Attention, stable = stable)
            elif attn_type == 'sparse':
                attn_class = SparseAttention
            elif attn_type == 'axial_row':
                attn_class = partial(SparseAxialCausalAttention, seq_len = max_seq_len, axis = 0, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'axial_col':
                attn_class = partial(SparseAxialCausalAttention, seq_len = max_seq_len, axis = 1, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'conv_like':
                attn_class = partial(SparseConvCausalAttention, seq_len = max_seq_len, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'mlp':
                attn_class = partial(gMLPBlock, seq_len = max_seq_len)
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')


            if attn_type == 'conditioned_i2t':
                attn = attn_class(dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            elif attn_type != 'conditioned_i2t' and attn_type != 'mlp':
                attn = attn_class(dim, causal = causal, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            else:
                attn = attn_class(dim = dim, causal = causal, dim_ff = dim * 4)
                
                
            ff = FeedForward(dim, mult = ff_mult, dropout = ff_dropout)

            if shift_tokens:
                attn, ff = map(lambda t: PreShiftToken(t, image_size = image_fmap_size, seq_len = max_seq_len), (attn, ff))

            layers.append(nn.ModuleList([
                LayerScale(dim, ind + 1, PreNorm(dim, attn)),
                LayerScale(dim, ind + 1, PreNorm(dim, ff))
            ]))


        ##! Executation Type
        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn, 'rotary_pos_emb': route_attn}

        self.layers = execute_type(layers, args_route = attn_route_map)
        # generate positional embeddings for rotary


        ##! Rotary Embedding
        pos_emb = None
        if rotary_emb:
            assert 'mlp' not in attn_types, 'you cannot use gMLPs if rotary embedding is turned on'

            rot_dim = dim_head // 3

            text_pos_emb = RotaryEmbedding(dim = rot_dim)
            img_axial_pos_emb = RotaryEmbedding(dim = rot_dim, freqs_for = 'pixel')

            text_freqs = text_pos_emb(torch.arange(num_tokens))
            img_to_text_freqs = text_pos_emb(torch.full((num_img_tokens,), 8192)) # image is given a position far away from text
            text_freqs = torch.cat((text_freqs, img_to_text_freqs), dim = 0)

            img_freqs_axial = img_axial_pos_emb(torch.linspace(-1, 1, steps = image_fmap_size))
            img_freqs = broadcat((rearrange(img_freqs_axial, 'i d -> i () d'), rearrange(img_freqs_axial, 'j d -> () j d')), dim = -1)
            img_freqs = rearrange(img_freqs, 'h w d -> (h w) d')

            text_axial_freqs = img_axial_pos_emb(torch.full((num_tokens,), -10.))  # text is given a position of -10 apart from the image axial positions, which is from range [-1, 1]
            text_axial_freqs = torch.cat((text_axial_freqs, text_axial_freqs), dim = -1)
            img_freqs = torch.cat((text_axial_freqs, img_freqs), dim = 0)

            pos_emb = torch.cat((text_freqs, img_freqs), dim = -1)
            pos_emb = rearrange(pos_emb[:-1], 'n d -> () () n d')

        # self.register_buffer('pos_emb', pos_emb)


    def forward(self, images, texts, **kwargs):
      
        #!# CNN
        images = self.cnn(images) # b, 256, dim (2048)
        images = self.img_embeddings(images)    # b, 256, dim (768)
        # print("images.shape after cnn: ",  images.shape)
        
        b, n_img, img_dim, device = *images.shape, images.device # b: batch_size, n_img: image의 tot_seq_len
        b, n_txt, device = *texts.shape, texts.device # b: batch_size, n_txt: text의 tot_seq_len
        # n = n_img + n_txt
        # assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        # assert n_img == self.condition_len, f'image length {n_img} must be equal to the condition length {self.condition_len}'

        with autocast():
            #!# img; token and positional embeddings
            if self.condition_len == 0:
                x_img = torch.empty(b, n_img, self.dim).to(device)
            else:
                # x_img = self.image_token_emb(images) # -> [B, tot_img_len, dim]
                x_img = images
                
                outs = []
                for x_img_slot in x_img.chunk(self.max_img_num, dim=1):  # x_img_slot: [B, img_len, dim]
                    out = self.image_pos_emb(x_img_slot) # out: [B, img_len, dim]
                    outs.append(out)
                x_img_pos = torch.cat(outs, dim = 1) # -> [B, tot_img_len, dim]
                x_img += x_img_pos
            
            #!# text; token and positional embeddings
            x_text = self.token_emb(texts)  # [B, text_seq_len] -> [B, text_seq_len, dim]
            x_text += self.pos_emb(x_text)   # [B, text_seq_len, dim]에 [1, text_seq_len, dim]이 더해짐
            x_text = F.dropout(x_text, p=self.emb_dropout, training=self.training)
            ## x_text : B x T x C -> T x B x C
            x = torch.cat((x_img, x_text), dim=1)    # -> [B, seq_len, dim] 
            # x = x.transpose(0, 1)
            
            #!# Into the layers
            x = self.layers(x, rotary_pos_emb = None if not self.rotary_emb else self.pos_emb, **kwargs)
            # x = x.transpose(0, 1)
            
            # norm and to logits
            x = self.norm(x)
            if exists(self.to_out):
                return self.to_out(x)  # -> [B, seq_len, num_tokens]
            else:
                return x