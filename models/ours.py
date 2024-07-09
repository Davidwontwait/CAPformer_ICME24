import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
import cv2

from models.snr_transformer import SNR_Bottleneck

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

# helpers

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

def dwconv_nxn_bn(inp,  kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, 1, bias=False,groups=inp),
        nn.BatchNorm2d(inp),
        nn.SiLU()
    )

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        #self.to_mask = nn.Linear(dim, inner_dim, bias=False)  ####1205

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(2)
            #print(mask.shape, 'Dots',dots.shape,q.shape)
            dots = dots.masked_fill(mask == 0, -1e9)


        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x,mask=None):
        for attn, ff in self.layers:
            x = attn(x,mask) + x
            x = ff(x) + x
        return x


class OurBlock(nn.Module):
    def __init__(self, dim=256, depth=2, channel=64, kernel_size=3, patch_size=2, mlp_dim=512, dropout=0.,heads=4,dim_head=8):
        super().__init__()
        self.p = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x,mask=None):

        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.p, pw=self.p)
        mask = rearrange(mask, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.p, pw=self.p)
        x = self.transformer(x,mask)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.p, w=w//self.p, ph=self.p, pw=self.p)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self,channel=64,n_layers=3,depth=2,patch_size=2,mlp_dim=512,cover=0.001,heads=4,dim_head=8,dim=128):
        super(Bottleneck, self).__init__()
        self.cover = cover
        self.layer_stack = nn.ModuleList([
            OurBlock(dim=dim,channel=channel,depth=1,patch_size=patch_size,mlp_dim=mlp_dim,heads=heads,dim_head=dim_head)
            for _ in range(n_layers)])


    def forward(self,x,mask=None):
        self.attn_map = []
        b, c, h_feature, w_feature = x.shape
        if mask is not  None:
            mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')
            mask = torch.mean(mask, dim=1, keepdim=True)
            mask[mask <= self.cover] = 0.0
            #mask = mask.repeat(1,c*2,1,1)
            self.attn_map.append(mask)

        for layer in self.layer_stack:
            x = layer(x,mask) + x
            self.attn_map.append(x)
        return x
#
import numpy as np

###############################
class ours(nn.Module):
    def __init__(self, opts):
        super(ours, self).__init__()
        nf = opts['nf']
        RBs = opts['nb']

        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, RBs)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, RBs)
        
        #self.conv_mask = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        #self.feature_extraction_mask = make_layer(ResidualBlock_noBN_f, 2)

        self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_4 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)




        self.upconv0 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)


        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        self.long_short = Bottleneck(channel = nf,n_layers = opts['n_layers'], depth= opts['depth'],
                                     patch_size=opts['patch_size'],mlp_dim=opts['mlp_dim'],cover=opts['cover'],
                                     dim_head=opts['dim_head'],heads=opts['heads'],dim=opts['dim'])
        if opts['mode'] == 'SNR':
            self.long_short = SNR_Bottleneck(channel=nf, n_layers=opts['n_layers'],
                                         patch_size=opts['patch_size'], threshold=opts['threshold'],
                                         dim_head=opts['dim_head'], heads=opts['heads'],dim=opts['dim'])
    def get_fea_map(self):
        return  self.long_short.attn_map

    def get_threshold(self):
        return self.long_short.get_threshold()


    def snr_map(self,x):
        dark = x
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114

        nf = torch.zeros_like(x)
        i = 0
        for img_LQ in x:
            img_nf = img_LQ.permute(1, 2, 0).detach().cpu().numpy() * 255.0
            img_nf = cv2.blur(img_nf, (5, 5))
            img_nf = img_nf * 1.0 / 255.0
            nf[i] = torch.Tensor(img_nf).float().permute(2, 0, 1).to(x.device)
            i += 1

        light = nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)
        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()

        return mask

    def forward(self, x):
        b, _, img_h, img_w = x.shape
        img_h_32 = int(32 * np.ceil(img_h / 32.0))
        img_w_32 = int(32 * np.ceil(img_w / 32.0))
        x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')


        #  calculate the mask
        mask = torch.mean(x, dim=1, keepdim=True)
        mask = self.snr_map(x)


        x_center = x

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))
        L1_fea_4 = self.lrelu(self.conv_first_4(L1_fea_3))

        fea = self.feature_extraction(L1_fea_4)

        fea = self.long_short(fea,mask)

        out_noise = self.recon_trunk(fea)

        out_noise = torch.cat([out_noise, L1_fea_4],dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv0(out_noise)))

        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))

        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))

        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))


        out_noise = self.conv_last(out_noise)

        # out_noise = out_noise + x
        out_noise = out_noise  + x_center


        return out_noise[:, :, :img_h, :img_w]


###############################
class ours1(nn.Module):
    def __init__(self, opts):
        super(ours1, self).__init__()
        RBs = opts['nb']
        self.dim = opts['nf']
        level = 3 #opts['level']

        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=self.dim)

        self.conv_first_1 = nn.Conv2d(3, self.dim, 3, 1, 1, bias=True)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = self.dim
        for i in range(self.level):
            self.encoder_layers.append(nn.ModuleList([
                SNR_Bottleneck(channel=dim_level, n_layers=1, depth=1,
                               patch_size=opts['patch_size'], mlp_dim=opts['mlp_dim'], cover=opts['cover'],
                               dim_head=opts['dim_head'], heads=opts['heads'], dim=opts['dim']),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
            ]))
            dim_level *= 2

        self.bottleneck = SNR_Bottleneck(channel=dim_level, n_layers=opts['n_layers'], depth=opts['depth'],
                                         patch_size=opts['patch_size'], mlp_dim=opts['mlp_dim'], cover=opts['cover'],
                                         dim_head=opts['dim_head'], heads=opts['heads'],dim=opts['dim'])




        self.feature_extraction = make_layer(ResidualBlock_noBN_f, RBs)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, RBs)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                SNR_Bottleneck(
                    dim=dim_level // 2, num_blocks=1, dim_head=self.dim,
                    heads=(dim_level // 2) // self.dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, 3, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def get_fea_map(self):
        return  self.long_short.attn_map

    def forward(self, x, mask=None):
        b, _, img_h, img_w = x.shape
        img_h_32 = int(32 * np.ceil(img_h / 32.0))
        img_w_32 = int(32 * np.ceil(img_w / 32.0))
        x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

        x_center = x

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))
        L1_fea_4 = self.lrelu(self.conv_first_4(L1_fea_3))

        fea = self.feature_extraction(L1_fea_4)

        fea = self.long_short(fea,mask)

        out_noise = self.recon_trunk(fea)

        out_noise = torch.cat([out_noise, L1_fea_4],dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv0(out_noise)))

        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))

        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))

        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))


        out_noise = self.conv_last(out_noise)

        # out_noise = out_noise + x
        out_noise = out_noise  + x_center


        return out_noise[:, :, :img_h, :img_w]


