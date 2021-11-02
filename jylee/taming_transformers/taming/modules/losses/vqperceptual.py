import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):  # 보통 threshold=30001
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))  # D는 real에 대해 logit값을 크게(+1이상) 만들려고 함.
    loss_fake = torch.mean(F.relu(1. + logits_fake))  # D는 fake에 대해 logit값을 작게(-1이하) 만들려고 함.
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):  # {'disc_conditional': False, 'disc_in_channels': 3, 'disc_start': 30001, 'disc_weight': 0.8, 'codebook_weight': 1.0}
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight      # 1.0
        self.pixel_weight = pixelloss_weight        # 1.0
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight  # 1.0

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, # 3
                                                 n_layers=disc_num_layers,  # 3
                                                 use_actnorm=use_actnorm,   # False
                                                 ndf=disc_ndf               # 64
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start  # 30001
        if disc_loss == "hinge":  # default가 hinge
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor            # 1.0
        self.discriminator_weight = disc_weight   # 0.8
        self.disc_conditional = disc_conditional  # False

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None): # nll_loss: recon_loss(tensor(scalar)), g_loss: generator_loss(tensor(scalar))  # last_layer: 디코더의 마지막 Conv의 Parameter([3, 128, 3, 3])
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0] # nll_loss을 last_layer로 미분한 값. nll_grads[3, 128, 3, 3] # retain_graph=True로 해야 nll_loss을 만드는 과정에서 생긴 computational graph가 안 사라짐.
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]     # g_loss을 last_layer로 미분한 값.g_grads[3, 128, 3, 3]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)  # 즉, last_layer가 g_loss에 미치는 영향이 작으면 d_weight을 키운다
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach() # detach()함으로써 d_weight을 통해 grad전파는 막는다
        d_weight = d_weight * self.discriminator_weight     # self.discriminator_weight = 0.8
        return d_weight  # tensor(scalar)

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,     # codebook_loss: tensor(scalar)       inputs, reconstructions: [B, 3, 256, 256]
                global_step, last_layer=None, cond=None, split="train"):         # last_layer = Parameter[3, 128, 3, 3]
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) # [3, 3, 256, 256]
        if self.perceptual_weight > 0:   # 보통 self.perceptual_weight = 1.0
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()) # -> [B, 1, 1, 1]
            rec_loss = rec_loss + self.perceptual_weight * p_loss   # [B, 3, 256, 256] + 1.0 * [B, 1, 1, 1]
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)  # [B, 3, 256, 256] -> tensor(scalar)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:  # cond는 항상 None
                assert not self.disc_conditional   # 보통 self.disc_conditional = False
                logits_fake = self.discriminator(reconstructions.contiguous())  # [B, 3, 256, 256] -> [B, 1, 30, 30]
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)  # generator(ae)입장에서는 logits_fake를 max해야 해서 -를 붙였다.

            try:  # train_step에서만 d_weight을 계산하고, validation_step에서는 except으로 간다
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)  # tensor(scalar)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start) # disc_factor = float.   self.discriminator_iter_start=30001. global_step이 threshold까지는 disc_factor=0, threshold이후는 disc_factor=1.0
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()    # 즉, 학습 처음부터 g_loss를 흘리지 않겠다. 30001step이후부터 흘리겠다.
            # loss: tensor(scalar)
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),  # rec_loss를 mean쳐서 넣으면 nll_loss이랑 똑같음
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log  # loss: tensor(scalar),   log: dict

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:  # cond는 항상 None
                logits_real = self.discriminator(inputs.contiguous().detach())           # [B, 3, 256, 256] -> [B, 1, 30, 30]
                logits_fake = self.discriminator(reconstructions.contiguous().detach())  # [B, 3, 256, 256] -> [B, 1, 30, 30]
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)  # global_step이 30001 step 이후부터 disc_factor = 1.0
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)                                     # 그 이후부터 d_loss를 흘리겠다.

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
