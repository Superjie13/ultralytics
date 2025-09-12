import torch
import torch.nn as nn
import torch.nn.functional as F

class ReIDPEBlock(nn.Module):
    """
    ReID feature projection + (optional) PE fusion
    - Execution order (when PE is enabled): PE fusion -> ReID residual branch
    - DDP friendly: no PE submodules are created when enable_pe=False
    - Fusion modes: 'add' | 'concat' | 'gated'
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        enable_pe: bool = False,  # whether to enable positional encoding
        pe_dim: int = 96,
        fusion_mode: str = "gated",      # 'add' | 'concat' | 'gated'
        cross_gate: bool = False,

        norm: str = "gn",                # 'gn' | 'bn' | 'none'
        use_se: bool = True,
    ):
        super().__init__()
        assert fusion_mode in ("add", "concat", "gated")
        self.out_dim = out_dim
        # self.res_scale = res_scale
        self.res_scale_pe = 1.
        self.use_se = use_se
        # self.l2norm = l2norm
        self.fusion_mode = fusion_mode
        self.cross_gate = cross_gate
        self.enable_pe = bool(enable_pe)

        def make_norm(ch):
            if norm == "gn": return nn.GroupNorm(1, ch)
            if norm == "bn": return nn.BatchNorm2d(ch)
            return nn.Identity()

        # -------- ReID path (always present) --------
        mid_ch = in_dim // 2 if (in_dim // 2) > out_dim else out_dim
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=True),
            nn.GELU(),
            make_norm(out_dim),
        )
        
        # self.proj2 = nn.Sequential(
        #     nn.Conv2d(out_dim, out_dim, 1, bias=True),
        #     make_norm(out_dim),
        #     nn.GELU(),
        # )

        # self.fuse = nn.Conv2d(pe_dim, out_dim, 1, bias=False)

        # -------- PE path (created only when enable_pe=True) --------
        if self.enable_pe:
            self.pe_proj = nn.Sequential(
                nn.Conv2d(pe_dim, out_dim, 1, bias=True),
                make_norm(out_dim),
            )

            if fusion_mode == "gated":
                self.pe_gate = None
                # self.pe_gate = nn.Conv2d(pe_dim * 2 if self.cross_gate else pe_dim, pe_dim, 1, bias=True)
                self.pe_concat_fuse = None
            elif fusion_mode == "concat":
                self.pe_concat_fuse = nn.Conv2d(mid_ch * 2, mid_ch, 1, bias=True)
                self.pe_gate = None
            else:  # 'add'
                self.pe_gate = None
                self.pe_concat_fuse = None


    @torch.no_grad()
    def _resize_to(self, x: torch.Tensor, size_hw):
        if x.shape[-2:] != size_hw:
            return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return x

    def forward(
        self,
        x: torch.Tensor,               # [B, Cin, H, W]
        pe: torch.Tensor = None,       # [B, Cpe, H?, W?]
        mask: torch.Tensor = None,     # [B, 1, H, W]
        use_pe: bool = True,
    ) -> torch.Tensor:
        B, _, H, W = x.shape

        base = self.proj1(x)  # [B,C,H,W]

        # 2) (optional) perform PE fusion first: base -> F0
        do_pe = self.enable_pe and use_pe and (pe is not None)
        if do_pe:
            pe = pe.to(dtype=base.dtype, device=base.device)
            pe = self._resize_to(pe, (H, W))
            P = self.pe_proj(pe)            # [B,C,H,W]
            if mask is not None:
                m = mask.to(dtype=P.dtype, device=P.device)
                P = P * m

            if self.fusion_mode == "add":
                F0 = base + self.res_scale_pe * P
            elif self.fusion_mode == "concat":
                fused = self.pe_concat_fuse(torch.cat([base, P], dim=1))
                if mask is not None:
                    fused = fused * m
                F0 = base + self.res_scale_pe * fused
            else:  # 'gated'
                # gate_in = torch.cat([base, P], dim=1) if self.cross_gate else P
                gate = torch.sigmoid(P)
                if mask is not None:
                    gate = gate * m
                F0 = base + self.res_scale_pe * (gate * P)
        else:
            F0 = base

        # 3) Then go through the ReID residual branch: F0 -> F_out
        # feats = [F0, self.dw1(F0)]
        # if self.dw2 is not None:
        #     feats.append(self.dw2(F0))
        # y = torch.cat(feats, dim=1)
        # y = self.fuse(y)                # initial 0
        # y = self.drop(y)
        # if self.use_se:
        #     w = F.adaptive_avg_pool2d(y, 1)
        #     w = F.relu(self.se_fc1(w), inplace=True)
        #     w = torch.sigmoid(self.se_fc2(w))
        #     y = y * w
        # F_out = F0 + self.feat_scale * y

        # y = self.proj2(F0)
        # y = F0
        # F_out = self.fuse(y)

        F_out = F0

        return F_out
