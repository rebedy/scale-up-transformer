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
from transformer_dalle.favor import FAVOR

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
        self.image_token_emb = nn.Embedding(num_img_tokens, dim, padding_idx)
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(image_fmap_size, image_fmap_size))
        
        # REPORT embedding and positioncal encoding
        self.token_emb = nn.Embedding(num_tokens, dim, padding_idx)  # num_tokens = text vocab size
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len) # =img_len * max_img_num + max_text_len
        self.layer_pos_emb = Always(None)
        
        self.dropout = nn.Dropout(emb_dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

        layers = nn.ModuleList([])
        
        ##! Get attention function and Set layers and Ready
        self.attn_type= attn_types
        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        for ind, attn_type in zip(range(depth), attn_type_layer):
            if attn_type == 'conditioned_i2t':  ###pure transformer에 대한 attention
                attn_class = partial(ConditionedCausalAttention_i2t, 
                                     seq_len = max_seq_len, condition_len=condition_len, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'favor+': ###performer에 대한 attention
                attn_class =  partial(FAVOR, 
                                   condition_len=condition_len,
                                   nb_features=nb_features, 
                                   generalized_attention = generalized_attention, 
                                   kernel_fn = kernel_fn, 
                                   no_projection = no_projection, 
                                   profile=profile
                                   )
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')
            
            if attn_type == 'conditioned_i2t':
                attn = attn_class(dim, seq_len = max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            elif attn_type == 'favor+':
                attn = attn_class(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            
            ff = FeedForward(dim, mult = ff_mult, dropout = ff_dropout)

            # if shift_tokens:
            #     attn, ff = map(lambda t: PreShiftToken(t, image_size = image_fmap_size, seq_len = max_seq_len), (attn, ff))

            layers.append(nn.ModuleList([
                LayerScale(dim, ind + 1, PreNorm(dim, attn)),
                LayerScale(dim, ind + 1, PreNorm(dim, ff))
            ]))

        
        ##! Executation Type
        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn, 'rotary_pos_emb': route_attn}

        self.layers = execute_type(layers, args_route = attn_route_map)
        # self.register_buffer('pos_emb', pos_emb)


    def forward(self, images, texts, **kwargs):
        b, n_img, device = *images.shape, images.device # b: batch_size, n_img: image의 tot_seq_len
        b, n_txt, device = *texts.shape, texts.device # b: batch_size, n_txt: text의 tot_seq_len
        n = n_img + n_txt
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        assert n_img == self.condition_len, f'image length {n_img} must be equal to the condition length {self.condition_len}'

        with autocast():
            #!# img; token and positional embeddings
            if self.condition_len == 0:
                print("self.condition_len == 0:!!!!!!!!!!!!!!!!!!!!!!!EMPTY!!!!!")
                exit()
                x_img = torch.empty(b, n_img, self.dim).to(device)
            else:
                x_img = self.image_token_emb(images) # -> [B, tot_img_len, dim]
                
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
            if self.attn_type == 'favor+':
                x = self.layers(x, pos_emb = self.pos_emb, **kwargs)
            else:
                x = self.layers(x, rotary_pos_emb = None, **kwargs)
                
            # x = x.transpose(0, 1)
            
            # norm and to logits
            x = self.norm(x)
            if exists(self.to_out):
                return self.to_out(x)  # -> [B, seq_len, num_tokens]
            else:
                return x
            


    @torch.no_grad()
    @eval_decorator
    def generate_text(
        self,
        images,
        texts,
        sos_token_idx,
        *,
        clip = None,
        mask = None,
        filter_thres = 0.5,
        temperature = 1.,
        img = None,
        num_init_img_tokens = None
    ):
        b, n_img, device = *images.shape, images.device # b: batch_size, n_img: image의 tot_seq_len
        b, n_txt, device = *texts.shape, texts.device # b: batch_size, n_txt: text의 tot_seq_len
        n = n_img + n_txt
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        assert n_img == self.condition_len, f'image length {n_img} must be equal to the condition length {self.condition_len}'

        images = texts[:, :self.condition_len] # make sure text is within bounds
        start_tok = torch.tensor([[sos_token_idx]]*b).to(device)
        out = torch.cat((images, start_tok), -1)

        # if exists(img):
        #     image_size = vae.image_size
        #     assert img.shape[1] == 3 and img.shape[2] == image_size and img.shape[3] == image_size, f'input image must have the correct image size {image_size}'

        #     indices = vae.get_codebook_indices(img)
        #     num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))  # OpenAI used 14 * 32 initial tokens to prime
        #     assert num_img_tokens < image_seq_len, 'number of initial image tokens for priming must be less than the total image token sequence length'

        #     indices = indices[:, :num_img_tokens]
        #     out = torch.cat((out, indices), dim = -1)

        for cur_len in range(out.shape[1], self.max_seq_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self(text, image, mask = mask)[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)

            sample -= (text_seq_len if is_image else 0) # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample), dim=-1)

            if out.shape[1] <= text_seq_len:
                mask = F.pad(mask, (0, 1), value = True)

        text_seq = out[:, :text_seq_len]

        if exists(clip):
            scores = clip(text_seq, images, return_loss = False)
            return images, scores

        return images