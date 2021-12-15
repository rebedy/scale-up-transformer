import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import time
import os
from functools import partial

from transformer_pytorch.FAVOR import FAVORAttention, ProjectionUpdater
from transformer_pytorch.transformer_attention import TransformerAttention
from transformer_pytorch.model_utils import *
from axial_positional_embedding import AxialPositionalEmbedding

def eval_decorator(fn):
    def inner(model, *args, **kwargs):  # args: (text[:1]), kwargs: {'filter_thres': 0.9}
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs) # generated image: [1, 3, 256, 256]
        model.train(was_training)  # model.training을 was_training값으로 바꿈
        return out
    return inner


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation = None):
        super().__init__()

        activation = default(activation, nn.GELU)

        self.w1 = nn.Linear(dim, mult*dim)
        self.act = activation()
        self.w2 = nn.Linear(mult*dim, dim)
        self.dropout = dropout

    def forward(self, x, **kwargs):
        out = self.w1(x)
        out = self.act(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.w2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            local_attn_heads = 0,
            causal = True,
            condition_len = 0,
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn = nn.ReLU(),
            ff_mult = 4,
            nb_features = None,
            feature_redraw_interval = 1000,
            use_scalenorm = False,
            use_rezero = False,
            ff_dropout = 0.,
            attn_dropout = 0.,
            cross_attend = False,
            auto_check_redraw = True,
            qkv_bias = True,
            attn_out_bias = True,
            no_projection = False,
            FAVOR = False
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)  # eg. (4, )
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads  # eg. (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4)
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:  # 보통 use_scalenorm = False
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:   # 보통 use_rezero = False
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)  # eg. dim=512

        if FAVOR:
            for _, local_heads in zip(range(depth), local_attn_heads):
                # Self-Attention + Feed Forward 합치는 부분
                layers.append(nn.ModuleList([
                    wrapper_fn(FAVORAttention(dim = dim, causal=causal, condition_len=condition_len, attn_type=attn_type, 
                                              generalized_attention=generalized_attention,
                                              kernel_fn = kernel_fn,
                                              heads=heads, local_heads=local_heads, nb_features=nb_features, 
                                              dropout=attn_dropout, no_projection=no_projection, 
                                              qkv_bias=qkv_bias, attn_out_bias=attn_out_bias)),
                    wrapper_fn(PositionWiseFeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, activation=None))
                ]))

                if not cross_attend:
                    continue
        else:
            for _, local_heads in zip(range(depth), local_attn_heads):
                layers.append(nn.ModuleList([
                    wrapper_fn(TransformerAttention(dim=dim, causal=causal, condition_len=condition_len, heads=heads,
                                local_heads = local_heads, dropout=attn_dropout, qkv_bias=qkv_bias, attn_out_bias=attn_out_bias)),
                    wrapper_fn(PositionWiseFeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, activation=None))
                ]))

                if not cross_attend:
                    continue

        execute_type = SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)  # len(): 2*depth if cross_attend else depth
        route_context = ((False, False), (True, False)) * depth  # eg. depth = 12 => len(route_context) = 24. 나의 경우 쓸모 없음
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context,
                             'context_mask': route_context} if cross_attend else {}  # cross attention 안 하는 거면 {}
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map})  # eg. ReversibleSequence으로 감싸진 모델

        # keeping track of when to redraw projections for all attention layers
        if FAVOR:
            self.auto_check_redraw = auto_check_redraw  # auto_check_redraw = True
            self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)
        else:
            self.auto_check_redraw = False

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x,
                **kwargs):  # x: [B, seq_len, dim]   kwargs = {'pos_emb': [1, seq_len, head_dim], 'mask': [B, seq_len]}
        if self.auto_check_redraw:  # auto_check_redraw = True
            self.proj_updater.redraw_projections()  # calls_since_last_redraw가 interval을 넘어가면 redraw  ## TODO: 얘가 뭐지?
        return self.net(x, **kwargs)  # -> [B, seq_len, dim]


class TransformerLM_i2t(nn.Module):
    def __init__(
            self,
            *,
            num_tokens, # text vocab size
            num_img_tokens, # img vocab size + num img pad
            max_seq_len,  # total max len; img_len * max_img_num + max_text_len
            max_img_num,  # num img slot
            dim,
            depth,
            heads=8,
            local_attn_heads=0,
            causal = True,
            condition_len = 0,
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn = nn.ReLU(),
            ff_mult=4,
            nb_features = None,
            feature_redraw_interval = 1000,
            reversible = False,
            emb_dropout = 0.,
            ff_dropout = 0.,
            attn_dropout = 0.,
            use_scalenorm=False,
            use_rezero=False,
            cross_attend=False,
            no_projection=False,
            tie_embed=False,
            rotary_position_emb=True,
            axial_position_emb=False,
            axial_position_shape=None,
            auto_check_redraw=True,
            qkv_bias=False,
            attn_out_bias=False,
            img_fmap_size=0,
            FAVOR=False,
            **kwargs
    ):
        super().__init__()
        # breakpoint()
        self.max_seq_len = max_seq_len
        self.max_img_num = max_img_num
        self.condition_len = condition_len
        local_attn_heads = cast_tuple(local_attn_heads)
        self.dim = dim
        dim_head = dim // heads

        # img
        if condition_len != 0:    # 이미지를 condition으로 준다면
            self.image_token_emb = nn.Embedding(num_img_tokens, dim)
            self.image_pos_emb = AxialPositionalEmbedding(dim=dim, axial_shape=(img_fmap_size, img_fmap_size))

        # text
        self.token_emb = nn.Embedding(num_tokens, dim)
        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)  # 항상 None값을 내뱉는다
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, local_attn_heads, causal, condition_len, attn_type, generalized_attention, kernel_fn,
                                       ff_mult, nb_features, feature_redraw_interval, use_scalenorm, use_rezero, 
                                       ff_dropout, attn_dropout, cross_attend, auto_check_redraw, 
                                       qkv_bias, attn_out_bias, no_projection, FAVOR)

        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) # if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, images, texts, return_encodings = False, **kwargs): # kwargs = {'mask': tensor with same shape x}
        b, n_img, device = *images.shape, images.device
        b, n_txt, device = *texts.shape, texts.device
        n = n_img + n_txt

        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        assert n_img == self.condition_len, f'image length {n_img} must be equal to the condition length {self.condition_len}'

        # img, token and positional embeddings
        if self.condition_len == 0:
            x_img = torch.empty(b, n_img, self.dim).to(device)
        else:
            x_img = self.image_token_emb(images)  # [B, tot_img_len, dim]
            outs = []
            for x_img_slot in x_img.chunk(self.max_img_num, dim=1):
                out = self.image_pos_emb(x_img_slot)  # out: [B, img_len, dim]
                outs.append(out)
            x_img_pos = torch.cat(outs, dim=1)  # -> [B, tot_img_len, dim]
            x_img += x_img_pos

        # text
            # text; token and positional embeddings
            x_text = self.token_emb(texts)  # [B, text_seq_len] -> [B, text_seq_len, dim]
            x_text += self.pos_emb(x_text)  # [B, text_seq_len, dim]에 [1, text_seq_len, dim]이 더해짐

            # merge
            x = torch.cat((x_img, x_text), dim=1)  # -> [B, seq_len, dim]

            x = self.dropout(x)

            # performer layers

            layer_pos_emb = self.layer_pos_emb(x)  # [1, seq_len, dim_head]  # TODO: rotary pos emb를 쓴다면 text에 대해서만 적용되어야 함. 가능한가?
            x = self.transformer(x, pos_emb=layer_pos_emb, **kwargs)  # x: [B, seq_len, dim] -> [B, seq_len, dim]

            # norm and to logits
            x = self.norm(x)

            if return_encodings:  # 보통 False
                return x

            if exists(self.to_out):
                return self.to_out(x)  # -> [B, seq_len, num_tokens]

            return x @ self.token_emb.weight.t()  # weight tieing했을 시

    @torch.no_grad()
    @eval_decorator
    def generate_texts(
            self,
            images,  # tensor[B, tot_img_len]
            *,
            sos_token_idx=None,
            eos_token_idx=None,
            pad_token_idx=None,
            filter_logits_fn='top_k',
            filter_thres=0.9,
            temperature=1.,
    ):
        if filter_logits_fn == 'top_k':
            filter_logits_fn = top_k
        elif filter_logits_fn == 'top_p':
            filter_logits_fn = top_p
        else:
            raise ValueError('filter_logits_fn must be in (top_k, top_p)')

        B, image_seq_len, device = *images.shape, images.device
        total_len = self.max_seq_len

        # [B, image_seq_len+1]에서 시작. 점점 길어질 것임.
        out = torch.cat(
            (images, torch.tensor([[sos_token_idx]] * B).to(device)),
            dim=-1
        )

        for cur_len in range(image_seq_len + 1, total_len):

            image, text = out[:, :image_seq_len], out[:, image_seq_len:]

            logits = self(image, text)[:, -1, :]  # -> logits: [B, num_text_tokens]
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)  # [B, num_text_tokens]
            sample = torch.multinomial(probs, 1)  # [B, 1]

            out = torch.cat((out, sample), dim=-1)

            # break check
            if ((out[:, image_seq_len:] == eos_token_idx).sum(dim=-1) > 0).sum() == B:
                break

        text_seq = out[:, image_seq_len:]  # [B, <text_seq_len]

        # postprocess
        indices = [list(row).index(eos_token_idx) if eos_token_idx in row else -1 for row in text_seq]
        for row, idx in enumerate(indices):
            if idx >= 0:
                text_seq[row, idx + 1:] = pad_token_idx
        return text_seq


class TransformerLM_Protein(nn.Module):
    def __init__(self,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 dim_head=64,
                 local_attn_heads=0,
                 local_window_size=256,
                 causal=False,
                 condition_len=0,
                 ff_mult=4,
                 nb_features=None,
                 feature_redraw_interval=1000,
                 reversible=False,
                 ff_chunks=1,
                 ff_glu=False,
                 emb_dropout=0.,
                 ff_dropout=0.,
                 attn_dropout=0.,
                 generalized_attention=False,
                 kernel_fn=nn.ReLU(),
                 use_scalenorm=False,
                 use_rezero=False,
                 cross_attend=False,
                 no_projection=False,
                 tie_embed=False,
                 rotary_position_emb=False,
                 axial_position_emb=False,
                 axial_position_shape=False,
                 auto_check_redraw=True,
                 qkv_bias=False,
                 attn_out_bias=False,
                 FAVOR=False,
                 **kwargs
                 ):
        super().__init__()
        self.max_seq_len = max_seq_len
        local_attn_heads = cast_tuple(local_attn_heads)
        self.dim = dim

        # input embedding
        self.token_emb = nn.Embedding(30, dim)
        self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, local_attn_heads, causal, condition_len, ff_mult, nb_features,
                                       feature_redraw_interval,
                                       reversible, use_scalenorm, use_rezero, ff_dropout, attn_dropout, cross_attend,
                                       auto_check_redraw, qkv_bias, attn_out_bias, no_projection, FAVOR)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, 30) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings=False, **kwargs):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)
        x = self.dropout(x)

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.transformer(x, pos_emb=layer_pos_emb, **kwargs)

        # norm
        x = self.norm(x)
        return self.to_out(x)

    @torch.no_grad()
    @eval_decorator
    def generate_proteins(self,
                          x,
                          filter_logits_fn='top_k',
                          filter_thres=0.9,
                          temperature=1.,
                          ):
        if filter_logits_fn == 'top_k':
            filter_logits_fn = top_k
        elif filter_logits_fn == 'top_p':
            filter_logits_fn = top_p
        else:
            raise ValueError('filter_logits_fn must be in (top_k, top_p)')

        out = x[:, 0].unsqueeze(-1)    # (B, 1) 첫번째 protein sequence만 제공

        for cur_len in range(self.max_seq_len-1):
            logits = self(out)[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
        return out

class TransformerLM_OneBillionWords(nn.Module):
    def __init__(self,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 dim_head=64,
                 local_attn_heads=0,
                 local_window_size=256,
                 causal=False,
                 condition_len=0,
                 ff_mult=4,
                 nb_features=None,
                 feature_redraw_interval=1000,
                 reversible=False,
                 ff_chunks=1,
                 ff_glu=False,
                 emb_dropout=0.,
                 ff_dropout=0.,
                 attn_dropout=0.,
                 generalized_attention=False,
                 kernel_fn=nn.ReLU(),
                 use_scalenorm=False,
                 use_rezero=False,
                 cross_attend=False,
                 no_projection=False,
                 tie_embed=False,
                 rotary_position_emb=False,
                 axial_position_emb=False,
                 axial_position_shape=False,
                 auto_check_redraw=True,
                 qkv_bias=False,
                 attn_out_bias=False,
                 FAVOR=False,
                 **kwargs
                 ):
        super().__init__()
        self.max_seq_len = max_seq_len
        local_attn_heads = cast_tuple(local_attn_heads)
        self.dim = dim

        # input embedding
        self.token_emb = nn.Embedding(30522, dim)
        self.positional_embedding = nn.Embedding(max_seq_len, dim)
        self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, local_attn_heads, causal, condition_len, ff_mult, nb_features,
                                       feature_redraw_interval,
                                       reversible, use_scalenorm, use_rezero, ff_dropout, attn_dropout, cross_attend,
                                       auto_check_redraw, qkv_bias, attn_out_bias, no_projection, FAVOR)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, 30522) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings=False, **kwargs):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)
        seq_len = torch.LongTensor([i for i in range(self.max_seq_len)]).cuda()
        seq_len = self.positional_embedding(seq_len)
        x = x + seq_len
        x = self.dropout(x)

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.transformer(x, pos_emb=layer_pos_emb, **kwargs)

        # norm
        x = self.norm(x)
        return self.to_out(x)
