import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.snr_transformer import SNR_Bottleneck
import functools
import numpy as np
####### From Retinexformer   framework architechture is followed SNR,but attention module is from Retinexformer  1203
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class BGMSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        # self.pos_emb = nn.Sequential(
        #     nn.Conv2d(dim_head * heads, dim, 3, 1, 1, bias=False, groups=dim),
        #     GELU(),
        #     nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        #     GELU(),
        #     nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        # )
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim_head*heads, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, mask=None):
        """
        x_in: [b,h,w,c]         # input_feature
        mask: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        # mask = mask.permute(0, 2, 3, 1)
        # d_mask = mask.shape[3]
        # d_v = self.num_heads*self.dim_head
        # mask = mask.repeat(1, 1, 1, d_v // d_mask)


        q, k, v, mask = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                            (q_inp, k_inp, v_inp, mask.flatten(1, 2)))


        # print(f'shape:{v.shape},{mask.shape}')
        v = v * mask
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        # out_p = self.pos_emb(v_inp.reshape(b, h, w, self.num_heads*self.dim_head).permute(
        #     0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        x = self.norm(x)
        out = self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return out


class EncoderLayer3(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, n_head, d_k,):
        super(EncoderLayer3, self).__init__()

        self.pos_ffn = FeedForward(dim=d_model)
        self.slf_attn = BGMSA(dim=d_model, dim_head=d_k, heads=n_head)

    def forward(self, x, mask):

        x = self.slf_attn(x,mask) + x
        x = self.pos_ffn(x) + x

        return x


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class myblock(nn.Module):
    def __init__(self, channel, dim=256, n_head=8, d_k=64):
        super().__init__()
        self.conv1 = conv_nxn_bn(channel, channel, 3)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, 3)

        self.transofmer = EncoderLayer3(d_model=dim,  n_head=n_head, d_k=d_k,)  # 1203

    def forward(self, x, mask=None):
        # input [b,c,h,w]
        y = x.clone()
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.permute(0,2,3,1)
        mask = mask.permute(0,2,3,1)
        # mask = F.interpolate(mask, size=[h, w], mode='nearest')
        x = self.transofmer(x, mask)
        x = x.permute(0,3,1,2)


        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x  # output [b,c,h,w]


class Bottleneck(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, channel=64, n_layers=6, dim=128, dim_head=64,heads=8):
        super().__init__()

        self.layer_stack = nn.ModuleList([
            myblock(channel=channel, n_head=heads, d_k=dim_head, dim=dim,)
            for _ in range(n_layers)])

    def forward(self, x, mask):
        ### input [b,c,h,w]
        b, c, h, w = x.shape
        self.attn_map = []

        self.attn_map.append(mask)

        x0 = x
        for layer in self.layer_stack:
            x0 = layer(x0, mask) + x0
            self.attn_map.append(x0)
        return x0


class Denoiser(nn.Module):
    def __init__(self, opts):
        super(Denoiser, self).__init__()
        dim = opts['dim']
        nf = opts['nf']
        nb = opts['nb']
        dim_head = opts['dim_head']
        heads = opts['heads']
        layers = opts['n_layers']
        p_size = opts['patch_size']
        self.level = opts['level']
        #mask
        self.bmap_estim = Illumination_Estimator(nf)

        # Input projection
        self.embedding = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        # self.embedding_mask = nn.Conv2d(1, nf, 3, 1, 1, bias=True)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        for i in range(self.level):
            self.encoder_layers.append(nn.ModuleList([
                nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
                nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            ]))

        # Bottleneck
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, nb)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, nb)
        # self.bottleneck = Bottleneck(channel=nf, dim=dim, dim_head=dim_head, heads=heads,n_layers=layers)
        self.bottleneck = SNR_Bottleneck(channel=nf,n_layers=layers, dim=dim, patch_size=p_size,dim_head=dim_head,heads=heads)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.level):
            self.decoder_layers.append(nn.ModuleList([
                nn.Conv2d(2 * nf, nf, 1, 1, bias=False),
                nn.ConvTranspose2d(nf, nf, stride=2,kernel_size=3, padding=1, output_padding=1),
            ]))

        # Output projection
        self.mapping = nn.Conv2d(nf, 3, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_fea_map(self):
        return  self.bottleneck.attn_map

    def forward(self, x,mask=None):
        """
        x:          [b,3,h,w]
        mask:       [b,1,h,w]
        return out: [b,3,h,w]
        """
        b, _, img_h, img_w = x.shape
        img_h_32 = int(32 * np.ceil(img_h / 32.0))
        img_w_32 = int(32 * np.ceil(img_w / 32.0))
        x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

        mask,b_map = self.bmap_estim(x)

        # mask = torch.mean(x, dim=1, keepdim=True)
        # mask = mask ** 0.8
        # mask = mask / (mask.max() - mask.min())

        input = x * b_map + x

        # Embedding
        fea = self.embedding(input)
        # if mask is None:

        # mask = self.embedding_mask(mask)

        # Encoder
        fea_encoder = []
        mask_list = []
        for (FeaDownSample, MaskDownsample) in self.encoder_layers:
            # mask_list.append(mask)
            fea_encoder.append(fea)
            fea = self.lrelu(FeaDownSample(fea))
            mask = self.lrelu(MaskDownsample(mask))

        # Bottleneck
        mask_list.append(mask)
        fea_encoder.append(fea)

        fea = self.feature_extraction(fea)
        fea = self.bottleneck(fea, mask)
        fea = self.recon_trunk(fea)

        # Decoder
        for i, (Fution,FeaUpSample, ) in enumerate(self.decoder_layers):
            fea = self.lrelu(Fution(torch.cat([fea, fea_encoder[self.level - i]], dim=1)))
            fea = self.lrelu(FeaUpSample(fea))

        # Mapping
        out = self.mapping(fea) + x

        return out[:, :, :img_h, :img_w]