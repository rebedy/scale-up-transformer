import torch
import torch.nn as nn
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# for routing arguments into the functions of the reversible layer
def route_args(router, args, depth):  # eg. router = {'mask': ((True, False), (True, False),...), 'pos_emb': ((True, False), (True, False),...)}  args = {'pos_emb': [1, seq_len, head_dim], 'mask': [1, seq_len]}   depth = 12
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]  # matched_keys =['mask', 'pos_emb']

    for key in matched_keys:
        val = args[key]  # eg. key = 'pos_emb', 'mask' 
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):    # f_args: {}  g_args: {}   routes: (True, False)   두번째 key에 대해서는 f_args: {'pos_emb': tensor[1, seq_len, head_dim]}로 비어있지 않겠지
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes) # new_f_args: {'pos_emb': tensor[1, seq_len, head_dim]}   new_g_args: {}
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args  # depth별로 router의 True인 위치에 {'pos_emb': tensor, 'mask': tensor} 삽입, False인 위치에는 {} 삽입
            # routed_args = [({...}, {}), ({...}, {}),...] ({f의 args}, {g의 args})

# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):  # args에 x[B, seq_len, dim]가 담겨있음.
        self.cpu_state = torch.get_rng_state()  # Returns the random number generator state as a torch.ByteTensor    tensor([164,  99,  25,  ...,   0,   0,   0]   len: 5056 
        if torch.cuda._initialized:   # True
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)  # [0], [tensor([255, 255, 255...rch.uint8)] 길이 816

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):  # args에 x[B, seq_len, dim]가 담겨있음. record_rng에 self.training를 받음
        if record_rng:
            self.record_rng(*args)

        if not set_rng: # set_rng = False
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args = {}, g_args = {}): # x: [B, seq_len, dim*2]  f_args: {'pos_emb': tensor, 'mask': tensor}  g_args: {}
        x1, x2 = torch.chunk(x, 2, dim=2)  # [B, seq_len, dim]  x1, x2는 동일함
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)  # f_args = {'pos_emb': tensor, 'mask': tensor}  # -> [B, seq_len, dim]
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)  # g_args = {}  # -> [B, seq_len, dim] 

        return torch.cat([y1, y2], dim=2)  # -> [B, seq_len, dim*2]

    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        # y1=x1+Attention(x2), y2=x2+FeedForward(y1) 에서 Attention가 f이고, FeedForward가 g임.
        y1, y2 = torch.chunk(y, 2, dim=2)    # [B, seq_len, dim*2] -> [B, seq_len, dim], [B, seq_len, dim]
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2) # [B, seq_len, dim*2] -> [B, seq_len, dim], [B, seq_len, dim]
        del dy

        with torch.enable_grad(): # Enables gradient calculation, if it has been disabled via no_grad or set_grad_enabled.
            y1.requires_grad = True
            # forward를 한 번 더 함! 메모리를 위해 희생되는 추가 연산
            gy1 = self.g(y1, set_rng=True, **g_args)  # y1=x1+Attention(x2), y2=x2+FeedForward(y1) 에서 Attention가 f이고, FeedForward가 g임.
            torch.autograd.backward(gy1, dy2)  # Computes the sum of gradients of given tensors with respect to graph leaves. # torch.autograd.backward(tensors, grad_tensors)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            # forward를 한 번 더 함! 메모리를 위해 희생되는 추가 연산
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, dx  # [B, seq_len, dim*2], [B, seq_len, dim*2] 

class _ReversibleFunction(Function): # PYTORCH: DEFINING NEW AUTOGRAD FUNCTION 참고하기
    @staticmethod  # @staticmethod: 인스턴스를 만들지 않아도 class의 메서드를 바로 실행할 수 있다
    def forward(ctx, x, blocks, args): # In the forward pass, we receive a Tensor containing the input and return a Tensor containing the output. ctx is a context object that can be used to stash information for backward computation. You can cache arbitrary objects for use in the backward pass using the ctx.save_for_backward method.
        ctx.args = args   # args: [({...}, {}), ({...}, {}),...]  ({f의 args}, {g의 args})
        for block, kwarg in zip(blocks, args):
            x = block(x, **kwarg)  # x: [B, seq_len, dim*2] -> [B, seq_len, dim*2] 근데 with torch.no_grad()으로 게산됨. 왜냐면 이 부분에 대한 gradient는 직접 구할 것이기에.
        ctx.y = x.detach()   # 아예 computational graph에서 떨어진 tensor를 y로 저장.
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):  # In the backward pass, we receive a Tensor containing the gradient of the loss with respect to the output, and we need to compute the gradient of the loss with respect to the input.
        y = ctx.y       # [B, seq_len, dim*2]  이 블록에서 나온 가장 마지막 output 
        args = ctx.args # [({...}, {}), ({...}, {}),...]  ({f의 args}, {g의 args})
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None

class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}):  # layers: len() = depth를 가지는 Modulelist. args_route: {'mask': ((True, False), (True, False),...), 'pos_emb': ((True, False), (True, False),..)}
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, **kwargs): # x: tensor[B, seq_len, dim]   kwargs: {'pos_emb': tensor[1, seq_len, dim_head], 'mask': tensor[B, seq_len]}
        args = route_args(self.args_route, kwargs, len(self.layers)) # eg. args_route: {'mask': ((True, False), (True, False),...), 'pos_emb': ((True, False), (True, False),...)}
        layers_and_args = list(zip(self.layers, args))               # args: [({...}, {}), ({...}, {}),...] ({f의 args}, {g의 args})

        for (f, g), (f_args, g_args) in layers_and_args:  # f: PreLayerNorm(SelfAttention)  g: PreLayerNorm(Chunk(FeedForward))  f_args: {'pos_emb': tensor[1, seq_len, dim_head], 'mask': tensor[B, seq_len]}  g_args: {}
            x = x + f(x, **f_args)  # -> [B, seq_len, dim]
            x = x + g(x, **g_args)  # -> [B, seq_len, dim]
        return x  # [B, seq_len, dim]

class ReversibleSequence(nn.Module):
    def __init__(self, blocks, args_route = {}): # eg. args_route: {'mask': ((True, False), (True, False),...), 'pos_emb': ((True, False), (True, False),...)}
        super().__init__()
        self.args_route = args_route
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g) for f, g in blocks]) # eg.  f: SelfAttention  g: Chunk(FeedForward)

    def forward(self, x, **kwargs):   # x: [B, seq_len, dim],   kwargs = {'pos_emb': [1, seq_len, head_dim], 'mask': [1, seq_len]}
        x = torch.cat([x, x], dim=-1) # -> [B, seq_len, 2*dim]

        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))  # eg. args_route: {'mask': ((True, False), (True, False),...), 'pos_emb': ((True, False), (True, False),...)}
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))  # args: [({...}, {}), ({...}, {}),...] ({f의 args}, {g의 args})

        out =  _ReversibleFunction.apply(x, blocks, args)   # -> [B, seq_len, dim*2] # To apply our Function, we use Function.apply method. We alias this as 'out'.
        return torch.stack(out.chunk(2, dim=-1)).sum(dim=0) # -> stack해서 [2, B, seq_len, dim] -> sum해서 [B, seq_len, dim]  y1=x1+Attention(x2), y2=x2+FeedForward(y1)에서 y1과 y2를 sum하는 것임.
