import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder, VUNet
from taming.modules.vqvae.quantize import VectorQuantizer


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,   # 보통 {'double_z': False, 'z_channels': 256, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [16], 'dropout': 0.0}
                 lossconfig, # 보통 {'target': 'taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator', 'params': {'disc_conditional': False, 'disc_in_channels': 3, 'disc_start': 30001, 'disc_weight': 0.8, 'codebook_weight': 1.0}}
                 n_embed,    # 보통 1024
                 embed_dim,  # 보통 256
                 ckpt_path=None,        # 보통 None 
                 ignore_keys=[],        # 보통 []
                 image_key="image",     # 반드시 "image". batch = { "image": torch(B, 256, 256, 3) value -1.0 ~ 1.0  ,  "file_path_": ['data/ffhq/00141.png', 'data/ffhq/00137.png',...] }이여서 그렇다.
                 colorize_nlabels=None, # 보통 None 
                 monitor=None           # 보통 None
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)    # {'target': 'taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator', 'params': {'disc_conditional': False, 'disc_in_channels': 3, 'disc_start': 30001, 'disc_weight': 0.8, 'codebook_weight': 1.0}}
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)   # n_embed = 1024, embed_dim = 256
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)      # self.quantize에 들어가기 위해 차원을 embed_dim로 바꿔주는 과정.
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1) # decode할 때 사용됨
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):   # x: [B, 3, 256, 256] 
        h = self.encoder(x)                      # -> [B, z_channels, 16, 16]  eg.[B, 256, 16, 16]
        h = self.quant_conv(h)                   # -> [B, embed_dim, 16, 16]   eg.[B, 256, 16, 16]
        quant, emb_loss, info = self.quantize(h) # -> z_q, loss, (perplexity, min_encodings, min_encoding_indices)  # [B, 256, 16, 16], tensor(scalar), ( tensor(scalar), [B*16*16, 1024], [B*16*16, 1] )
        return quant, emb_loss, info             # [B, 256, 16, 16], tensor(scalar), ( tensor(scalar), [B*16*16, 1024], [B*16*16, 1] )

    def decode(self, quant):
        quant = self.post_quant_conv(quant)      # [B, embed_dim, 16, 16] -> [B, z_channels, 16, 16] eg. [B, 256, 16, 16]
        dec = self.decoder(quant)                # -> [B, 3, 256, 256]
        return dec  # [B, 3, 256, 256]

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)  # [B, 3, 256, 256] -> z_q, emb_loss, (perplexity, min_encodings, min_encoding_indices): [B, 256, 16, 16], tensor(scalar), ( tensor(scalar), [B*16*16, 1024], [B*16*16, 1] )
        dec = self.decode(quant)             # [B, 256, 16, 16] -> [B, 3, 256, 256]
        return dec, diff  # [B, 3, 256, 256], tensor(scalar)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)  # tensor[B, 256, 256, 3] -> [B, 3, 256, 256]  NOTE good practice: .to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx): # optimizer_idx 필수!  # batch = { "image": torch(B, 256, 256, 3) value -1.0 ~ 1.0  ,  "file_path_": ['data/ffhq/00141.png', 'data/ffhq/00137.png',...], 'class': torch[B]  }
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)  # x: [B, 3, 256, 256] -> xrec: [B, 3, 256, 256], qloss: tensor(scalar)
        # NOTE: 확인할 것: 같은 배치에 대해 optimizer_idx를 바꿔서 두 번 들어온다! 두 번 들어와야 self.global_step이 1증가함.
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,      # aeloss: tensor(scalar). global_step이 30001을 넘어가기 전까진 g_loss가 aeloss에 더해지지 않음
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss   # lightning에서는 training_step에서 loss만 return해주면 opt.zero_grad(), opt.backward(), opt.step()을 알아서 다 해줌.

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,  # discloss: tensor(scalar). global_step이 30001을 넘어가기 전까진 discloss(d_loss)가 0
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):  # batch = { "image": torch[B, 256, 256, 3] value -1.0 ~ 1.0  ,  "file_path_": ['data/ffhq/00141.png', 'data/ffhq/00137.png',...], 'class': torch[B] }
        x = self.get_input(batch, self.image_key) # self.image_key = 'image'
        xrec, qloss = self(x)  # x: [B, 3, 256, 256] -> xrec: [B, 3, 256, 256], qloss: tensor(scalar)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,      # optimizer_idx = 0   # aeloss: tensor(scalar)
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,  # optimizer_idx = 1   # discloss: tensor(scalar)
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True) # on_step=True이면 logger에 step마다 찍고 on_epoch=True이면 step마다 찍은 걸 평균 내서 에폭별로도 기록.
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True) # ifsync_dist = True, reduces the metric across GPUs/TPUs. sync_dist_op='mean'(default)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))  # DCGAN에서 제안한 베타
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []    # optimizer가 여러개일 경우 이런식으로 함. 뒤에 []은 lr_scheduler자리임

    def get_last_layer(self):
        return self.decoder.conv_out.weight  # Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)): [3, 128, 3, 3]

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)  # x: [B, 3, 256, 256]
        x = x.to(self.device)
        xrec, _ = self(x)    # xrec: [B, 3, 256, 256], qloss: tensor(scalar)
        if x.shape[1] > 3:   # x.shape[1]=3이라 여기 스킵
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer
