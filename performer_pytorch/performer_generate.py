import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager

from performer_pytorch.performer_pytorch import *
from axial_positional_embedding import AxialPositionalEmbedding

from sampling import *

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers

def eval_decorator(fn):
    def inner(model, *args, **kwargs):  # args: (text[:1]), kwargs: {'filter_thres': 0.9}
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs) # generated image: [1, 3, 256, 256]
        model.train(was_training)  # model.training을 was_training값으로 바꿈
        return out
    return inner


class PerformerLM_i2t(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,     # text vocab size
        num_img_tokens, # img vocab size + num img pad 
        max_seq_len,    # total max len; img_len * max_img_num + max_text_len
        max_img_num,    # num img slot
        dim,
        depth,
        heads,
        dim_head = 64,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = True,
        condition_len = 0,
        ff_mult = 4,
        nb_features = 256,
        feature_redraw_interval = 1000,
        reversible = True,
        ff_chunks = 10,
        ff_glu = True,
        emb_dropout = 0.1,
        ff_dropout = 0.1,
        attn_dropout = 0.1,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,
        rotary_position_emb = False,
        axial_position_emb = False,
        axial_position_shape = None,
        auto_check_redraw = True,
        qkv_bias = False,
        attn_out_bias = False,
        img_fmap_size = 0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_img_num = max_img_num
        self.condition_len = condition_len
        local_attn_heads = cast_tuple(local_attn_heads)
        
        # img;
        self.image_token_emb = nn.Embedding(num_img_tokens, dim)
        self.image_pos_emb = AxialPositionalEmbedding(dim=dim, axial_shape=(img_fmap_size, img_fmap_size))
        
        # text;
        self.token_emb = nn.Embedding(num_tokens, dim)   # num_tokens = text vocab size
        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, condition_len, ff_mult, 
            nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, 
            ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias, attn_out_bias)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, images, texts, return_encodings = False, **kwargs):  # kwargs = {'mask': tensor with same shape x}
        b, n_img, device = *images.shape, images.device # b: batch_size, n_img: image의 tot_seq_len
        b, n_txt, device = *texts.shape, texts.device # b: batch_size, n_txt: text의 tot_seq_len
        n = n_img + n_txt
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        assert n_img == self.condition_len, f'image length {n_img} must be equal to the condition length {self.condition_len}'

        # img; token and positional embeddings
        x_img = self.image_token_emb(images) # -> [B, tot_img_len, dim]
        outs = []
        for x_img_slot in x_img.chunk(self.max_img_num, dim=1):  # x_img_slot: [B, img_len, dim]
            out = self.image_pos_emb(x_img_slot) # out: [B, img_len, dim]
            outs.append(out)
        x_img_pos = torch.cat(outs, dim = 1) # -> [B, tot_img_len, dim]
        x_img += x_img_pos

        # text; token and positional embeddings
        x_text = self.token_emb(texts)  # [B, text_seq_len] -> [B, text_seq_len, dim]
        x_text += self.pos_emb(x_text)   # [B, text_seq_len, dim]에 [1, text_seq_len, dim]이 더해짐

        # merge
        x = torch.cat((x_img, x_text), dim=1)    # -> [B, seq_len, dim]

        x = self.dropout(x)

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)  # [1, seq_len, dim_head]  # TODO: rotary pos emb를 쓴다면 text에 대해서만 적용되어야 함. 가능한가?
        x = self.performer(x, pos_emb = layer_pos_emb, **kwargs) # x: [B, seq_len, dim] -> [B, seq_len, dim]

        # norm and to logits
        x = self.norm(x)

        if return_encodings:   # 보통 False
            return x

        if exists(self.to_out):
            return self.to_out(x)  # -> [B, seq_len, num_tokens]

        return x @ self.token_emb.weight.t()  # weight tieing했을 시
    
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        x,
        *,
        sos_token_idx = None,
        eos_token_idx = None,
        pad_token_idx = None,
        num_beams=1,
        do_sample=True,
        filter_logits_fn = 'both',
        top_k = 50,
        top_p = 0.9,
        temperature = 1.,
        ):
        
        length_penalty=2
        early_stopping=False
        
        self.fix_projection_matrices_()
        
        b, img_seq_len, device = *x.shape, x.device

        start_tok = torch.tensor([[sos_token_idx]]*b).to(device)
        out = torch.cat((x, start_tok), -1)
        logits = torch.empty_like(out)
        
        if num_beams <= 1:
            for cur_len in range(img_seq_len+1, self.max_seq_len):
                image, text = out[:, :img_seq_len], out[:, img_seq_len:]
                with torch.no_grad():
                    logits = self(image, text)
                logit = logits[:, -1, :] # -> logits: [B, num_text_tokens]
                
                if do_sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        logit = logit / temperature #sharpness
                        
                    # Top-p/top-k sampling
                    if filter_logits_fn.lower() in ['top_k','topk', 'k']:
                        filtered_logits = top_k_sampling(logit, thres=top_k)
                    elif filter_logits_fn.lower() in ['top_p', 'topp', 'p']:
                        filtered_logits = top_p_sampling(logit, thres=top_p)
                    elif filter_logits_fn.lower() in ['both', 'all', 'top_k_top_p', 'topktopp',]:
                        filtered_logits = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
                    else:
                        assert filter_logits_fn is None, "Please assign the function to sample logis among ['both, 'top_p', 'top_k']."
                        filtered_logits=1
                
                    probs = F.softmax(filtered_logits, dim = -1) # [B, num_text_tokens]
                    sample = torch.multinomial(probs, 1) # [B, 1]
                    
                else:
                    # Greedy sampling
                    sample = torch.argmax(logits, dim=-1)
                
                out = torch.cat((out, sample), dim=-1)
                
                # break check
                if ( (out[:, img_seq_len:] == eos_token_idx).sum(dim=-1) > 0 ).sum() == b:
                    break
                
        else:
            # generated hypotheses
            generated_hyps = [BeamHypotheses(num_beams, self.max_seq_len, length_penalty, early_stopping=early_stopping)
            for _ in range(b)]   
            # scores for each sentence in the beam
            beam_scores = torch.zeros((b, num_beams), dtype=torch.float, device=device)

            # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
            if do_sample is False:
                beam_scores[:, 1:] = -1e9
                beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)
                
            # cache compute states  -> not encoder decoder
            past = None
                
                
            for cur_len in range(img_seq_len+1, self.max_seq_len):
                image, text = out[:, :img_seq_len], out[:, img_seq_len:]
                with torch.no_grad():
                    logits = self(image, text)
                logit = logits[:, -1, :] # -> logits: [B, num_text_tokens]
            
            pass  ####Beam Search!

                
        gen_out = out[:, img_seq_len:]  # this will be less than the max_text_len
        logits_out = logits[:, img_seq_len:]
        
        # postprocess
        indices = [list(row).index(eos_token_idx) if eos_token_idx in row else -1 for row in gen_out]
        for row, idx in enumerate(indices):
            if idx >= 0:
                gen_out[row, idx+1:] = pad_token_idx
                
    
        return gen_out, logits_out
    
    
     
     
     
     

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret