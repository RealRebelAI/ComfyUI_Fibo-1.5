# Native-ish Fibo transformer runtime for ComfyUI GGUF.
# Keeps Diffusers/BRIA tensor names so the GGUF keys can load directly.
# Upstream Fibo reference: BriaFiboTransformer2DModel (CC-BY-NC-4.0).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.model_base
import comfy.conds
import comfy.latent_formats
import comfy.supported_models_base

print("[Fibo] fibo_model v5-48ch loaded")

def timestep_embedding(t, dim=256, max_period=10000):
    half=dim//2
    freqs=torch.exp(-math.log(max_period)*torch.arange(half,device=t.device,dtype=torch.float32)/half)
    a=t.float()[:,None]*freqs[None]
    e=torch.cat([torch.cos(a),torch.sin(a)],dim=-1)
    if dim%2: e=torch.cat([e,torch.zeros_like(e[:,:1])],dim=-1)
    return e

def rope1d(pos, dim, theta=10000.0):
    inv=1.0/(theta**(torch.arange(0,dim,2,device=pos.device,dtype=torch.float32)/dim))
    a=pos.float()[:,None]*inv[None]
    return torch.repeat_interleave(torch.cos(a),2,-1),torch.repeat_interleave(torch.sin(a),2,-1)

def make_rope(ids, axes=(16,56,56), theta=10000.0):
    cs=[]; ss=[]
    for i,d in enumerate(axes):
        c,s=rope1d(ids[:,i],d,theta); cs.append(c); ss.append(s)
    return torch.cat(cs,-1),torch.cat(ss,-1)

def apply_rope(x, rope):
    c,s=rope
    c=c.to(x)[None,:,None,:]; s=s.to(x)[None,:,None,:]
    xe=x[...,0::2]; xo=x[...,1::2]
    out=torch.empty_like(x)
    out[...,0::2]=xe*c[...,0::2]-xo*s[...,0::2]
    out[...,1::2]=xo*c[...,1::2]+xe*s[...,0::2]
    return out

class TimeEmbed(nn.Module):
    def __init__(self,dim,operations,device=None,dtype=None):
        super().__init__()
        self.timestep_embedder=nn.Module()
        self.timestep_embedder.linear_1=operations.Linear(256,dim,bias=True,device=device,dtype=dtype)
        self.timestep_embedder.linear_2=operations.Linear(dim,dim,bias=True,device=device,dtype=dtype)
    def forward(self,t,dtype):
        x=timestep_embedding(t,256).to(dtype=dtype)
        return self.timestep_embedder.linear_2(F.silu(self.timestep_embedder.linear_1(x)))

class AdaLNZero(nn.Module):
    def __init__(self,dim,operations,device=None,dtype=None):
        super().__init__(); self.linear=operations.Linear(dim,dim*6,bias=True,device=device,dtype=dtype)
    def forward(self,x,e):
        sm,sc,ga,sh,ss,gm=self.linear(F.silu(e)).chunk(6,1)
        y=F.layer_norm(x,(x.shape[-1],),eps=1e-6)
        y=y*(1+sc[:,None])+sm[:,None]
        return y,ga,sh,ss,gm

class AdaLNZeroSingle(nn.Module):
    def __init__(self,dim,operations,device=None,dtype=None):
        super().__init__(); self.linear=operations.Linear(dim,dim*3,bias=True,device=device,dtype=dtype)
    def forward(self,x,e):
        sh,sc,g=self.linear(F.silu(e)).chunk(3,1)
        y=F.layer_norm(x,(x.shape[-1],),eps=1e-6)
        return y*(1+sc[:,None])+sh[:,None],g

class AdaLNContinuous(nn.Module):
    def __init__(self,dim,operations,device=None,dtype=None):
        super().__init__(); self.linear=operations.Linear(dim,dim*2,bias=True,device=device,dtype=dtype)
    def forward(self,x,e):
        sc,sh=self.linear(F.silu(e)).chunk(2,1)
        y=F.layer_norm(x,(x.shape[-1],),eps=1e-6)
        return y*(1+sc[:,None])+sh[:,None]

class FiboAttention(nn.Module):
    def __init__(self,dim,heads,hd,operations,added=False,pre_only=False,device=None,dtype=None):
        super().__init__()
        self.heads=heads; self.hd=hd; self.added=added; self.pre_only=pre_only
        self.to_q=operations.Linear(dim,dim,bias=True,device=device,dtype=dtype)
        self.to_k=operations.Linear(dim,dim,bias=True,device=device,dtype=dtype)
        self.to_v=operations.Linear(dim,dim,bias=True,device=device,dtype=dtype)
        self.norm_q=nn.RMSNorm(hd,eps=1e-6,device=device,dtype=dtype)
        self.norm_k=nn.RMSNorm(hd,eps=1e-6,device=device,dtype=dtype)
        if not pre_only:
            self.to_out=nn.ModuleList([operations.Linear(dim,dim,bias=True,device=device,dtype=dtype),nn.Identity()])
        if added:
            self.add_q_proj=operations.Linear(dim,dim,bias=True,device=device,dtype=dtype)
            self.add_k_proj=operations.Linear(dim,dim,bias=True,device=device,dtype=dtype)
            self.add_v_proj=operations.Linear(dim,dim,bias=True,device=device,dtype=dtype)
            self.norm_added_q=nn.RMSNorm(hd,eps=1e-6,device=device,dtype=dtype)
            self.norm_added_k=nn.RMSNorm(hd,eps=1e-6,device=device,dtype=dtype)
            self.to_add_out=operations.Linear(dim,dim,bias=True,device=device,dtype=dtype)
    def split(self,x): return x.view(x.shape[0],x.shape[1],self.heads,self.hd)
    def forward(self,x,ctx=None,rope=None,mask=None):
        q=self.norm_q(self.split(self.to_q(x))); k=self.norm_k(self.split(self.to_k(x))); v=self.split(self.to_v(x))
        nctx=0
        if ctx is not None:
            eq=self.norm_added_q(self.split(self.add_q_proj(ctx))); ek=self.norm_added_k(self.split(self.add_k_proj(ctx))); ev=self.split(self.add_v_proj(ctx))
            nctx=ctx.shape[1]; q=torch.cat([eq,q],1); k=torch.cat([ek,k],1); v=torch.cat([ev,v],1)
        if rope is not None: q=apply_rope(q,rope); k=apply_rope(k,rope)
        q=q.transpose(1,2); k=k.transpose(1,2); v=v.transpose(1,2)
        if mask is not None:
            if mask.ndim==2: mask=mask[:,None,None,:]
            mask=mask.to(device=q.device)
        o=F.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2)
        o=o.reshape(o.shape[0],o.shape[1],self.heads*self.hd)
        if ctx is not None:
            oc,oi=o[:,:nctx],o[:,nctx:]
            return self.to_out[0](oi),self.to_add_out(oc)
        return o

class FF(nn.Module):
    def __init__(self,dim,operations,device=None,dtype=None):
        super().__init__()
        p=nn.Module(); p.proj=operations.Linear(dim,dim*4,bias=True,device=device,dtype=dtype)
        self.net=nn.ModuleList([p,nn.Identity(),operations.Linear(dim*4,dim,bias=True,device=device,dtype=dtype)])
    def forward(self,x): return self.net[2](F.gelu(self.net[0].proj(x),approximate="tanh"))

class CaptionProjection(nn.Module):
    def __init__(self,ind,outd,operations,device=None,dtype=None):
        super().__init__(); self.linear=operations.Linear(ind,outd,bias=False,device=device,dtype=dtype)
    def forward(self,x): return self.linear(x)

class DualBlock(nn.Module):
    def __init__(self,dim,heads,hd,operations,device=None,dtype=None):
        super().__init__()
        self.norm1=AdaLNZero(dim,operations,device,dtype); self.norm1_context=AdaLNZero(dim,operations,device,dtype)
        self.attn=FiboAttention(dim,heads,hd,operations,True,False,device,dtype)
        self.norm2=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6,device=device,dtype=dtype)
        self.ff=FF(dim,operations,device,dtype)
        self.norm2_context=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6,device=device,dtype=dtype)
        self.ff_context=FF(dim,operations,device,dtype)
    def forward(self,h,c,e,rope,mask=None):
        nh,ga,sh,sc,gm=self.norm1(h,e); nc,cga,csh,csc,cgm=self.norm1_context(c,e)
        ai,ac=self.attn(nh,nc,rope,mask)
        h=h+ga[:,None]*ai
        h2=self.norm2(h)*(1+sc[:,None])+sh[:,None]
        h=h+gm[:,None]*self.ff(h2)
        c=c+cga[:,None]*ac
        c2=self.norm2_context(c)*(1+csc[:,None])+csh[:,None]
        c=c+cgm[:,None]*self.ff_context(c2)
        return c,h

class SingleBlock(nn.Module):
    def __init__(self,dim,heads,hd,operations,device=None,dtype=None):
        super().__init__()
        self.norm=AdaLNZeroSingle(dim,operations,device,dtype)
        self.proj_mlp=operations.Linear(dim,dim*4,bias=True,device=device,dtype=dtype)
        self.attn=FiboAttention(dim,heads,hd,operations,False,True,device,dtype)
        self.proj_out=operations.Linear(dim*5,dim,bias=True,device=device,dtype=dtype)
    def forward(self,x,e,rope,mask=None):
        r=x; n,g=self.norm(x,e)
        mlp=F.gelu(self.proj_mlp(n),approximate="tanh")
        a=self.attn(n,None,rope,mask)
        return r+g[:,None]*self.proj_out(torch.cat([a,mlp],-1))

class FiboTransformer(nn.Module):
    def __init__(self,in_channels=48,num_layers=8,num_single_layers=38,attention_head_dim=128,num_attention_heads=24,
                 joint_attention_dim=4096,text_encoder_dim=2048,axes_dims_rope=(16,56,56),rope_theta=10000,time_theta=10000,
                 device=None,dtype=None,operations=None,**kwargs):
        super().__init__()
        self.dtype=dtype or torch.float32; self.in_channels=in_channels
        self.inner_dim=num_attention_heads*attention_head_dim
        self.axes_dims_rope=tuple(axes_dims_rope); self.rope_theta=rope_theta
        self.time_embed=TimeEmbed(self.inner_dim,operations,device,dtype)
        self.context_embedder=operations.Linear(joint_attention_dim,self.inner_dim,bias=True,device=device,dtype=dtype)
        self.x_embedder=operations.Linear(in_channels,self.inner_dim,bias=True,device=device,dtype=dtype)
        self.transformer_blocks=nn.ModuleList([DualBlock(self.inner_dim,num_attention_heads,attention_head_dim,operations,device,dtype) for _ in range(num_layers)])
        self.single_transformer_blocks=nn.ModuleList([SingleBlock(self.inner_dim,num_attention_heads,attention_head_dim,operations,device,dtype) for _ in range(num_single_layers)])
        self.norm_out=AdaLNContinuous(self.inner_dim,operations,device,dtype)
        self.proj_out=operations.Linear(self.inner_dim,in_channels,bias=True,device=device,dtype=dtype)
        total=num_layers+num_single_layers
        self.caption_projection=nn.ModuleList([CaptionProjection(text_encoder_dim,self.inner_dim//2,operations,device,dtype) for _ in range(total)])
    def ids(self,tlen,h,w,device,dtype):
        txt=torch.zeros(tlen,3,device=device,dtype=dtype)
        img=torch.zeros(h,w,3,device=device,dtype=dtype)
        img[...,1]=torch.arange(h,device=device,dtype=dtype)[:,None]
        img[...,2]=torch.arange(w,device=device,dtype=dtype)[None,:]
        return torch.cat([txt,img.reshape(h*w,3)],0)
    def forward(self,x,timestep,context=None,fibo_text_layers=None,fibo_attention_mask=None,**kwargs):
        input_ndim=x.ndim
        frames=1

        if input_ndim==5:
            # Comfy Wan latent format: B,C,T,H,W.
            # Fibo image generation currently expects a single image frame.
            b,c,frames,h,w=x.shape
            if frames != 1:
                raise RuntimeError(
                    f"Fibo image runtime currently expects T=1 Wan latent, got shape {tuple(x.shape)}"
                )
            x=x[:,:,0].permute(0,2,3,1).reshape(b,h*w,c)

        elif input_ndim==4:
            # Standard Comfy image latent: B,C,H,W.
            b,c,h,w=x.shape
            x=x.permute(0,2,3,1).reshape(b,h*w,c)

        elif input_ndim==3:
            # Already packed sequence: B,S,C.
            b,s,c=x.shape
            h=int(math.sqrt(s))
            if h*h != s:
                raise RuntimeError(
                    f"Cannot infer square Fibo latent grid from sequence length {s}."
                )
            w=s//h

        else:
            raise RuntimeError(
                f"Unsupported Fibo latent rank {input_ndim}; expected BCHW, BCTHW, or BSC. "
                f"Got shape {tuple(x.shape)}"
            )

        if x.shape[-1] != self.in_channels:
            raise RuntimeError(
                f"Fibo expected {self.in_channels} latent channels after packing, "
                f"but received {x.shape[-1]}. Packed shape={tuple(x.shape)}"
            )

        if context is None or fibo_text_layers is None:
            raise RuntimeError("Fibo requires cross_attn and fibo_text_layers conditioning.")
        hidden=self.x_embedder(x)
        temb=self.time_embed(timestep.to(hidden.dtype),hidden.dtype)
        enc=self.context_embedder(context)
        rope=make_rope(self.ids(enc.shape[1],h,w,hidden.device,hidden.dtype),self.axes_dims_rope,self.rope_theta)
        total=len(self.transformer_blocks)+len(self.single_transformer_blocks)
        if fibo_text_layers.shape[1]<total:
            fibo_text_layers=torch.cat([fibo_text_layers,fibo_text_layers[:,-1:].repeat(1,total-fibo_text_layers.shape[1],1,1)],1)
        elif fibo_text_layers.shape[1]>total:
            fibo_text_layers=fibo_text_layers[:,-total:]
        proj=[self.caption_projection[i](fibo_text_layers[:,i]) for i in range(total)]
        mask=None
        if fibo_attention_mask is not None and not bool(torch.all(fibo_attention_mask)):
            im=torch.ones(fibo_attention_mask.shape[0],hidden.shape[1],device=fibo_attention_mask.device,dtype=fibo_attention_mask.dtype)
            mask=torch.cat([fibo_attention_mask,im],1).bool()[:,None,None,:]
        bi=0
        for block in self.transformer_blocks:
            enc=torch.cat([enc[:,:,:self.inner_dim//2],proj[bi]],-1); bi+=1
            enc,hidden=block(hidden,enc,temb,rope,mask)
        for block in self.single_transformer_blocks:
            enc=torch.cat([enc[:,:,:self.inner_dim//2],proj[bi]],-1); bi+=1
            n=enc.shape[1]
            joined=block(torch.cat([enc,hidden],1),temb,rope,mask)
            enc,hidden=joined[:,:n],joined[:,n:]
        out=self.proj_out(self.norm_out(hidden,temb))

        if input_ndim==5:
            out=out.reshape(b,h,w,self.in_channels).permute(0,3,1,2).contiguous()
            out=out.unsqueeze(2)  # B,C,1,H,W
        elif input_ndim==4:
            out=out.reshape(b,h,w,self.in_channels).permute(0,3,1,2).contiguous()

        return out

class FiboLatentFormat(comfy.latent_formats.LatentFormat):
    """Native Fibo 1.5 image latent format.

    Fibo uses a 48-channel 2D latent with 16x spatial downscale.
    Keep model-space latents pass-through; VAE normalization is handled
    by the Fibo VAE wrapper.
    """
    latent_channels = 48
    latent_dimensions = 2
    spacial_downscale_ratio = 16
    temporal_downscale_ratio = 1
    scale_factor = 1.0

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


class FiboModelConfig(comfy.supported_models_base.BASE):
    unet_extra_config={}
    latent_format=FiboLatentFormat
    supported_inference_dtypes=[torch.bfloat16,torch.float16,torch.float32]
    sampling_settings={"multiplier":1000.0,"shift":1.0}
    memory_usage_factor=2.0
    def model_type(self,state_dict,prefix=""): return comfy.model_base.ModelType.FLOW
    def get_model(self,state_dict,prefix="",device=None): return FiboBaseModel(self,device=device)

class FiboBaseModel(comfy.model_base.BaseModel):
    def __init__(self,model_config,device=None):
        super().__init__(model_config,model_type=comfy.model_base.ModelType.FLOW,device=device,unet_model=FiboTransformer)
    def extra_conds(self,**kwargs):
        out=super().extra_conds(**kwargs)
        if kwargs.get("cross_attn") is not None: out["c_crossattn"]=comfy.conds.CONDRegular(kwargs["cross_attn"])
        if kwargs.get("fibo_text_layers") is not None: out["fibo_text_layers"]=comfy.conds.CONDRegular(kwargs["fibo_text_layers"])
        if kwargs.get("fibo_attention_mask") is not None: out["fibo_attention_mask"]=comfy.conds.CONDRegular(kwargs["fibo_attention_mask"])
        return out
