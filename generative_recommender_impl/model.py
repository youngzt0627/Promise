import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 2e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x32 * rms
        return (out * self.weight.float()).to(x.dtype)


def apply_rope(x: Tensor, position_ids: Tensor, base: float = 4096.0) -> Tensor:
    if x.size(-1) != 64:
        raise ValueError(f"RoPE expects head_dim=64, got {x.size(-1)}")
    if position_ids.dim() != 2:
        raise ValueError(f"position_ids must be [B, L], got shape {tuple(position_ids.shape)}")

    x32 = x.float()
    position_ids = position_ids.to(device=x.device, dtype=torch.float32)
    d = x.size(-1)
    half = d // 2
    theta = base ** (-2.0 * torch.arange(half, device=x.device, dtype=torch.float32) / d)
    freq = position_ids[..., None] * theta[None, None, :]
    sin_half = torch.sin(freq)
    cos_half = torch.cos(freq)
    sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(1)
    cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(1)
    x1 = x32[..., :half]
    x2 = x32[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (x32 * cos + rot * sin).to(x.dtype)


class PerTokenGatedFFN(nn.Module):
    def __init__(
        self,
        d: int = 768,
        length: int = 5,
        expansion_factor: int = 4,
        inner_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d = d
        self.length = length
        self.expansion_factor = expansion_factor
        self.inner_dim = inner_dim or (d * expansion_factor)

        self.fc1_weight = nn.Parameter(torch.empty(length, d, self.inner_dim))
        self.gate_weight = nn.Parameter(torch.empty(length, d, self.inner_dim))
        self.fc2_weight = nn.Parameter(torch.empty(length, self.inner_dim, d))
        self.fc1_bias = nn.Parameter(torch.empty(length, self.inner_dim))
        self.gate_bias = nn.Parameter(torch.empty(length, self.inner_dim))
        self.fc2_bias = nn.Parameter(torch.empty(length, d))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.fc1_weight)
        nn.init.xavier_normal_(self.gate_weight)
        nn.init.xavier_normal_(self.fc2_weight)
        nn.init.zeros_(self.fc1_bias)
        nn.init.zeros_(self.gate_bias)
        nn.init.zeros_(self.fc2_bias)

    def forward(self, x: Tensor, start_index: int = 0) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"PerTokenGatedFFN expects [B, T, D], got {tuple(x.shape)}")

        batch_size, seq_len, dim = x.shape
        if dim != self.d:
            raise ValueError(f"Expected hidden dim {self.d}, got {dim}")
        if start_index < 0 or start_index + seq_len > self.length:
            raise ValueError(
                f"Invalid start_index={start_index} for seq_len={seq_len} and length={self.length}"
            )

        fc1_weight = self.fc1_weight[start_index : start_index + seq_len]
        gate_weight = self.gate_weight[start_index : start_index + seq_len]
        fc2_weight = self.fc2_weight[start_index : start_index + seq_len]
        fc1_bias = self.fc1_bias[start_index : start_index + seq_len]
        gate_bias = self.gate_bias[start_index : start_index + seq_len]
        fc2_bias = self.fc2_bias[start_index : start_index + seq_len]

        fc1_out = torch.einsum("btd,tdm->btm", x, fc1_weight) + fc1_bias.unsqueeze(0)
        gate_out = torch.einsum("btd,tdm->btm", x, gate_weight) + gate_bias.unsqueeze(0)
        hidden = fc1_out * F.silu(gate_out)
        out = torch.einsum("btm,tmd->btd", hidden, fc2_weight) + fc2_bias.unsqueeze(0)
        return out.view(batch_size, seq_len, dim)


class PrefixDiscriminator(nn.Module):
    def __init__(
        self,
        sid_vocab_sizes: Sequence[int],
        token_dim: int,
        num_levels: int = 3,
        mlp_hidden_dim: Optional[int] = None,
        loss_weights: Optional[Sequence[float]] = None,
        num_negative_samples: int = 10,
    ) -> None:
        super().__init__()
        if len(sid_vocab_sizes) != 3:
            raise ValueError(f"sid_vocab_sizes must have length 3, got {len(sid_vocab_sizes)}")
        if num_levels != 3:
            raise ValueError(f"PrefixDiscriminator currently requires num_levels=3, got {num_levels}")

        self.sid_vocab_sizes = [int(size) for size in sid_vocab_sizes]
        self.token_dim = int(token_dim)
        self.num_levels = int(num_levels)
        self.num_negative_samples = int(num_negative_samples)
        self.loss_weights = [float(x) for x in (loss_weights or [1.0, 1.0, 1.0])]
        if len(self.loss_weights) != 3:
            raise ValueError(f"loss_weights must have length 3, got {len(self.loss_weights)}")
        mlp_hidden = self.token_dim if mlp_hidden_dim is None else int(mlp_hidden_dim)

        self.sid_emb_tables = nn.ModuleList(
            [nn.Embedding(vocab_size, self.token_dim) for vocab_size in self.sid_vocab_sizes]
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * self.token_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.register_buffer("valid_prefixes_depth1", torch.zeros(0, 1, dtype=torch.long), persistent=False)
        self.register_buffer("valid_prefixes_depth2", torch.zeros(0, 2, dtype=torch.long), persistent=False)
        self.register_buffer("valid_prefixes_depth3", torch.zeros(0, 3, dtype=torch.long), persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for emb in self.sid_emb_tables:
            nn.init.uniform_(emb.weight, a=-0.05, b=0.05)

    def finite_check(self, x: Tensor) -> Tensor:
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    def set_valid_prefixes(self, valid_prefixes_by_depth: Sequence[Tensor]) -> None:
        if len(valid_prefixes_by_depth) != self.num_levels:
            raise ValueError(
                f"valid_prefixes_by_depth must have length {self.num_levels}, got {len(valid_prefixes_by_depth)}"
            )

        expected_widths = [1, 2, 3]
        buffer_names = ["valid_prefixes_depth1", "valid_prefixes_depth2", "valid_prefixes_depth3"]
        for depth, (prefixes, expected_width, buffer_name) in enumerate(
            zip(valid_prefixes_by_depth, expected_widths, buffer_names),
            start=1,
        ):
            prefixes = prefixes.long()
            if prefixes.dim() != 2 or prefixes.size(1) != expected_width:
                raise ValueError(
                    f"valid prefixes for depth {depth} must be [N, {expected_width}], got {tuple(prefixes.shape)}"
                )
            setattr(self, buffer_name, prefixes.clone())

    def _valid_prefix_pool(self, depth: int) -> Tensor:
        if depth == 1:
            return self.valid_prefixes_depth1
        if depth == 2:
            return self.valid_prefixes_depth2
        if depth == 3:
            return self.valid_prefixes_depth3
        raise ValueError(f"depth must be in [1, {self.num_levels}], got {depth}")

    def build_prefix_embedding(self, path_sids: Tensor, depth: int) -> Tensor:
        if depth < 1 or depth > self.num_levels:
            raise ValueError(f"depth must be in [1, {self.num_levels}], got {depth}")

        prefix_parts = []
        for level_idx in range(depth):
            sid_l = path_sids[..., level_idx].long().clamp(0, self.sid_vocab_sizes[level_idx] - 1)
            prefix_parts.append(self.sid_emb_tables[level_idx](sid_l))
        return torch.stack(prefix_parts, dim=-2).mean(dim=-2)

    def encode_user(
        self,
        input_emb: Tensor,
        input_mask: Optional[Tensor],
        detach_input: bool,
    ) -> Tensor:
        user_tokens = input_emb.detach() if detach_input else input_emb
        user_tokens = self.finite_check(user_tokens)
        if input_mask is None:
            return user_tokens.mean(dim=1)

        mask = input_mask.float().unsqueeze(-1)
        masked_sum = (user_tokens * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return self.finite_check(masked_sum / denom)

    def _score_prefix_embeddings(self, user_repr: Tensor, prefix_emb: Tensor) -> Tensor:
        if prefix_emb.dim() == 2:
            fused = torch.cat([user_repr, prefix_emb], dim=-1)
            return self.fuse_mlp(fused).squeeze(-1).float()
        if prefix_emb.dim() == 3:
            fused = torch.cat(
                [user_repr[:, None, :].expand(prefix_emb.size(0), prefix_emb.size(1), user_repr.size(-1)), prefix_emb],
                dim=-1,
            )
            return self.fuse_mlp(fused.reshape(-1, fused.size(-1))).reshape(prefix_emb.size(0), prefix_emb.size(1)).float()
        raise ValueError(f"prefix_emb must be rank 2 or 3, got {prefix_emb.dim()}")

    def _sample_negative_prefixes(self, pos_prefix: Tensor, sid_valid_row: Tensor, depth: int) -> Tensor:
        batch_size = pos_prefix.size(0)
        neg_prefixes = pos_prefix.new_zeros(batch_size, self.num_negative_samples, depth)
        valid_pool = self._valid_prefix_pool(depth)
        if valid_pool.numel() == 0:
            raise ValueError("valid prefix pool is empty; call set_valid_prefixes() before training")

        valid_pool = valid_pool.to(device=pos_prefix.device)
        for row_idx in range(batch_size):
            if not bool(sid_valid_row[row_idx].item()):
                continue
            candidate_mask = ~(valid_pool == pos_prefix[row_idx]).all(dim=-1)
            candidate_pool = valid_pool[candidate_mask]
            if candidate_pool.size(0) == 0:
                raise ValueError(f"depth {depth} valid prefix pool must contain at least one negative prefix")
            if candidate_pool.size(0) >= self.num_negative_samples:
                sample_idx = torch.randperm(candidate_pool.size(0), device=pos_prefix.device)[: self.num_negative_samples]
            else:
                sample_idx = torch.randint(
                    low=0,
                    high=candidate_pool.size(0),
                    size=(self.num_negative_samples,),
                    device=pos_prefix.device,
                )
            neg_prefixes[row_idx] = candidate_pool[sample_idx]
        return neg_prefixes

    def _score_train_depth(
        self,
        user_repr: Tensor,
        target_sid: Tensor,
        sid_valid_row: Tensor,
        depth: int,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        batch_size = target_sid.shape[0]
        prefix = target_sid[:, :depth]
        pos_prefix_emb = self.build_prefix_embedding(prefix, depth)
        neg_prefix = self._sample_negative_prefixes(prefix, sid_valid_row, depth)
        neg_prefix_emb = self.build_prefix_embedding(neg_prefix, depth)

        pos_logits = self._score_prefix_embeddings(user_repr, pos_prefix_emb).unsqueeze(1)
        neg_logits = self._score_prefix_embeddings(user_repr, neg_prefix_emb)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        logits = self.finite_check(logits)

        labels = torch.zeros(batch_size, device=target_sid.device, dtype=torch.long)
        ce = F.cross_entropy(logits, labels, reduction="none")
        weight = sid_valid_row.float()
        denom = weight.sum().clamp_min(1e-6)
        loss = (ce * weight).sum() / denom

        pred_idx = logits.argmax(dim=-1)
        acc = ((pred_idx == 0).float() * weight).sum() / denom
        pos_logit = logits[:, 0]
        rank = (logits > pos_logit[:, None]).float().sum(dim=1)
        metrics = {
            "acc": acc,
            "hr@1": ((rank < 1).float() * weight).sum() / denom,
            "hr@5": ((rank < 5).float() * weight).sum() / denom,
            "hr@10": ((rank < 10).float() * weight).sum() / denom,
            "hr@50": ((rank < 50).float() * weight).sum() / denom,
        }
        return loss, logits, metrics

    def forward(
        self,
        input_emb: Tensor,
        input_mask: Optional[Tensor],
        target_sid: Tensor,
        sid_valid_row: Tensor,
    ) -> Dict[str, Union[Tensor, List[Tensor], List[Dict[str, Tensor]]]]:
        user_repr = self.encode_user(input_emb, input_mask, detach_input=True)

        losses: List[Tensor] = []
        logits_by_depth: List[Tensor] = []
        metrics_by_depth: List[Dict[str, Tensor]] = []
        total_loss = torch.zeros((), device=input_emb.device, dtype=torch.float32)

        for depth in [1, 2, 3]:
            loss_d, logits_d, metrics_d = self._score_train_depth(
                user_repr=user_repr,
                target_sid=target_sid,
                sid_valid_row=sid_valid_row,
                depth=depth,
            )
            losses.append(loss_d)
            logits_by_depth.append(logits_d)
            metrics_by_depth.append(metrics_d)
            total_loss = total_loss + self.loss_weights[depth - 1] * loss_d

        return {
            "loss": total_loss,
            "loss_by_depth": losses,
            "logits_by_depth": logits_by_depth,
            "metrics_by_depth": metrics_by_depth,
        }

    def score_candidates(
        self,
        input_emb: Tensor,
        input_mask: Optional[Tensor],
        cand_sids: Tensor,
        depth: int,
    ) -> Tensor:
        user_repr = self.encode_user(input_emb, input_mask, detach_input=False)
        prefix_emb = self.build_prefix_embedding(cand_sids[:, :, :depth], depth)
        aux_scores = self._score_prefix_embeddings(user_repr, prefix_emb)
        return self.finite_check(aux_scores)


class GenerativeRecommender(nn.Module):
    hidden_size = 512
    num_decoder_layers = 5
    num_heads = 8
    head_dim = 64
    semantic_id_vocab_sizes = (256, 256, 256)
    num_special_tokens = 18
    total_vocab_size = 786
    semantic_part_ranges = ((18, 274), (274, 530), (530, 786))
    prefix_len = 21
    target_len = 3
    decoder_len = 24
    drop_last_sid_token = True
    rope_base = 4096.0
    projection_logit_scale = 20.0
    default_beam_sizes = (100, 300, 500)
    default_discriminator_pool_sizes = (200, 600, 1000)

    def __init__(self, d_in_enc: int = 512) -> None:
        super().__init__()
        self.d_in_enc = d_in_enc

        self.embedding_table = nn.Embedding(self.total_vocab_size, self.hidden_size)

        self.proj_in = nn.Linear(self.hidden_size, self.hidden_size)
        self.final_ln = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.ln_c = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size, eps=1e-6) for _ in range(self.num_decoder_layers)]
        )
        self.q_proj = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_decoder_layers)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_decoder_layers)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_decoder_layers)]
        )
        self.q_norm = nn.ModuleList(
            [RMSNorm(self.head_dim, eps=2e-5) for _ in range(self.num_decoder_layers)]
        )
        self.k_norm = nn.ModuleList(
            [RMSNorm(self.head_dim, eps=2e-5) for _ in range(self.num_decoder_layers)]
        )
        self.out_proj = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_decoder_layers)]
        )

        self.ln2 = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size, eps=1e-6) for _ in range(self.num_decoder_layers)]
        )
        self.ffn_layers = nn.ModuleList(
            [
                PerTokenGatedFFN(
                    d=self.hidden_size,
                    length=self.decoder_len,
                    expansion_factor=4,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )

        self.sid_output_kernels = nn.Parameter(
            torch.empty(self.target_len, self.hidden_size, self.semantic_id_vocab_sizes[0])
        )
        self.prefix_discriminator = PrefixDiscriminator(
            sid_vocab_sizes=self.semantic_id_vocab_sizes,
            token_dim=self.hidden_size,
            num_levels=self.target_len,
            mlp_hidden_dim=self.hidden_size,
            loss_weights=[1.0, 1.0, 1.0],
            num_negative_samples=10,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.embedding_table.weight, std=0.02)
        nn.init.xavier_normal_(self.sid_output_kernels)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        self.prefix_discriminator.reset_parameters()

    def _full_dec_mask(self, batch_size: int, length: int, device: torch.device) -> Tensor:
        return torch.ones(batch_size, length, device=device, dtype=torch.long)

    def _prepare_enc_mask(self, enc_mask_seq: Optional[Tensor], batch_size: int, seq_len: int, device: torch.device) -> Tensor:
        if enc_mask_seq is None:
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        return enc_mask_seq.to(device=device, dtype=torch.long)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _stabilize_scores(self, score: Tensor) -> Tensor:
        score = score.float()
        score = score - score.amax(dim=-1, keepdim=True)
        return torch.clamp(score, min=-1e4, max=20.0)

    def _self_attn_full(
        self,
        h: Tensor,
        layer_idx: int,
        dec_mask: Optional[Tensor],
        return_cache: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        batch_size, dec_len, _ = h.shape
        h4 = self.ln_c[layer_idx](h)
        q = self._reshape_heads(self.q_proj[layer_idx](h4))
        k = self._reshape_heads(self.k_proj[layer_idx](h4))
        v = self._reshape_heads(self.v_proj[layer_idx](h4))
        q = self.q_norm[layer_idx](q)
        k = self.k_norm[layer_idx](k)

        if dec_mask is not None:
            pos = torch.cumsum(dec_mask.long(), dim=1) - 1
            pos = torch.clamp(pos, min=0)
        else:
            pos = torch.arange(dec_len, device=h.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        q = apply_rope(q, pos, base=self.rope_base)
        k = apply_rope(k, pos, base=self.rope_base)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(dec_len, dec_len, device=h.device, dtype=torch.bool),
            diagonal=1,
        )
        score = score.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
        if dec_mask is not None:
            key_mask = dec_mask[:, None, None, :].to(score.dtype)
            score = score + (key_mask - 1.0) * 1e4
        score = self._stabilize_scores(score)
        attn = torch.softmax(score, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, dec_len, self.hidden_size)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = self.out_proj[layer_idx](out)
        h_out = h + out
        if return_cache:
            return h_out, (k, v)
        return h_out

    def self_attn_layer(self, h: Tensor, layer_idx: int, dec_mask: Optional[Tensor]) -> Tensor:
        return self._self_attn_full(h, layer_idx, dec_mask, return_cache=False)

    def ffn_layer(self, h: Tensor, layer_idx: int, start_index: int = 0) -> Tensor:
        h2 = self.ln2[layer_idx](h)
        ff = self.ffn_layers[layer_idx](h2, start_index=start_index)
        return h + ff

    def decode(
        self,
        x_dec: Tensor,
        dec_mask: Optional[Tensor],
    ) -> Tensor:
        h = self.proj_in(x_dec)
        for layer_idx in range(self.num_decoder_layers):
            h = self.self_attn_layer(h, layer_idx, dec_mask)
            h = self.ffn_layer(h, layer_idx, start_index=0)
        h = self.final_ln(h)
        return h

    def cosine_project(self, h_next: Tensor) -> Tensor:
        _, steps, dim = h_next.shape
        if dim != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {dim}")
        if steps != self.target_len:
            raise ValueError(f"cosine_project expects {self.target_len} steps, got {steps}")
        h32 = F.normalize(h_next.float(), p=2, dim=-1)
        w32 = F.normalize(self.sid_output_kernels.float(), p=2, dim=1)
        logits = torch.einsum("bsd,sdv->bsv", h32, w32)
        return logits * self.projection_logit_scale

    def project_step(self, h_last: Tensor, step_idx: int) -> Tensor:
        if h_last.dim() != 2 or h_last.size(-1) != self.hidden_size:
            raise ValueError(f"h_last must be [B, {self.hidden_size}], got {tuple(h_last.shape)}")
        if step_idx < 0 or step_idx >= self.target_len:
            raise ValueError(f"step_idx must be in [0, {self.target_len}), got {step_idx}")
        h32 = F.normalize(h_last.float(), p=2, dim=-1)
        w32 = F.normalize(self.sid_output_kernels[step_idx].float(), p=2, dim=0)
        logits = h32 @ w32
        return logits * self.projection_logit_scale

    def _build_decoder_inputs(
        self,
        prefix_emb: Tensor,
        query_ids: Tensor,
        prefix_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, prefix_len, hidden_size = prefix_emb.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"prefix_emb hidden size must be {self.hidden_size}, got {hidden_size}")
        query_emb = self.embedding_table(query_ids)
        x_dec = torch.cat([prefix_emb, query_emb], dim=1)
        if x_dec.size(1) > self.decoder_len:
            raise ValueError(f"Decoder sequence too long: {x_dec.size(1)} > {self.decoder_len}")
        if prefix_mask is None:
            prefix_mask = torch.ones(batch_size, prefix_len, device=x_dec.device, dtype=torch.long)
        else:
            prefix_mask = prefix_mask.to(device=x_dec.device, dtype=torch.long)
        query_mask = torch.ones(batch_size, query_ids.size(1), device=x_dec.device, dtype=torch.long)
        dec_mask = torch.cat([prefix_mask, query_mask], dim=1)
        return x_dec, dec_mask

    def forward(
        self,
        input_emb: Tensor,
        enc_mask_seq: Optional[Tensor],
        target_sid: Tensor,
    ) -> Dict[str, Tensor]:
        batch_size, seq_len, hidden_size = input_emb.shape
        if hidden_size != self.d_in_enc:
            raise ValueError(f"input_emb hidden size must be {self.d_in_enc}, got {hidden_size}")
        if target_sid.shape != (batch_size, 3):
            raise ValueError(f"target_sid must be [B, 3], got {tuple(target_sid.shape)}")

        device = input_emb.device
        enc_mask_seq = self._prepare_enc_mask(enc_mask_seq, batch_size, seq_len, device)

        vocab_sizes = torch.tensor(
            self.semantic_id_vocab_sizes,
            device=device,
            dtype=target_sid.dtype,
        )
        target_local = torch.maximum(target_sid.long(), torch.zeros_like(target_sid.long()))
        target_local = torch.minimum(target_local, (vocab_sizes - 1).unsqueeze(0))
        sid_valid_row = ((target_sid.long() >= 0) & (target_sid.long() < vocab_sizes.unsqueeze(0))).all(dim=1)
        sid0_global = self.semantic_part_ranges[0][0] + target_local[:, 0]
        sid1_global = self.semantic_part_ranges[1][0] + target_local[:, 1]
        bos_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        bos_ids = bos_ids.unsqueeze(1)
        sid_ids = torch.stack([sid0_global, sid1_global], dim=1)
        dec_ids = torch.cat([bos_ids, sid_ids], dim=1)
        x_dec, dec_mask = self._build_decoder_inputs(input_emb, dec_ids, enc_mask_seq)
        h_full = self.decode(x_dec, dec_mask)
        h_next = h_full[:, -self.target_len :, :]
        ntp_logits = self.cosine_project(h_next)

        logits2d = ntp_logits.reshape(batch_size * self.target_len, self.semantic_id_vocab_sizes[0])
        labels1d = target_local.reshape(batch_size * self.target_len)
        loss = F.cross_entropy(logits2d, labels1d, reduction="none")
        loss_stack = loss.view(batch_size, 3)
        pos_w = torch.ones(3, device=device, dtype=loss_stack.dtype)
        loss_sum = (loss_stack * pos_w.unsqueeze(0)).sum()
        denom = pos_w.sum() * batch_size
        denom = torch.clamp(denom, min=1.0)
        ntp_loss = loss_sum / denom
        discriminator_out = self.prefix_discriminator(
            input_emb=input_emb,
            input_mask=enc_mask_seq,
            target_sid=target_local,
            sid_valid_row=sid_valid_row,
        )
        total_loss = ntp_loss + discriminator_out["loss"]

        return {
            "loss": total_loss,
            "ntp_loss": ntp_loss,
            "ctr_logits": ntp_logits,
            "discriminator_loss": discriminator_out["loss"],
            "discriminator_loss_by_depth": discriminator_out["loss_by_depth"],
            "discriminator_logits_by_depth": discriminator_out["logits_by_depth"],
            "discriminator_metrics_by_depth": discriminator_out["metrics_by_depth"],
            "h_next": h_next,
            "h_full": h_full,
        }

    def _decode_query_ids(
        self,
        prefix_emb: Tensor,
        query_ids: Tensor,
        prefix_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x_dec, dec_mask = self._build_decoder_inputs(prefix_emb, query_ids, prefix_mask)
        return self.decode(x_dec, dec_mask)

    def _normalize_beam_sizes(
        self,
        beam_sizes: Union[int, Sequence[int]],
    ) -> List[int]:
        if isinstance(beam_sizes, int):
            return [beam_sizes, beam_sizes, beam_sizes]
        beam_list = list(beam_sizes)
        if not beam_list:
            raise ValueError("beam_sizes sequence cannot be empty")
        while len(beam_list) < 3:
            beam_list.append(beam_list[-1])
        return [int(k) for k in beam_list[:3]]

    def _gather_beam_tensor(self, source: Tensor, indices: Tensor) -> Tensor:
        if source.dim() == 2:
            return torch.gather(source, dim=1, index=indices)
        gather_index = indices.unsqueeze(-1).expand(*indices.shape, source.size(-1))
        return torch.gather(source, dim=1, index=gather_index)

    def _expand_prefix_inputs(
        self,
        prefix_inputs: Tuple[Tensor, Tensor],
        beam_size: int,
    ) -> Tuple[Tensor, Tensor]:
        prefix_emb, prefix_mask = prefix_inputs
        batch_size, length, dim = prefix_emb.shape
        prefix_emb = (
            prefix_emb.unsqueeze(1)
            .expand(batch_size, beam_size, length, dim)
            .reshape(batch_size * beam_size, length, dim)
        )
        prefix_mask = (
            prefix_mask.unsqueeze(1)
            .expand(batch_size, beam_size, prefix_mask.size(1))
            .reshape(batch_size * beam_size, prefix_mask.size(1))
        )
        return prefix_emb, prefix_mask

    def _slice_step_logits(self, decoder_out: Tensor, step_idx: int) -> Tensor:
        return self.project_step(decoder_out[:, -1, :], step_idx)

    def set_valid_prefixes(self, valid_prefixes_by_depth: Sequence[Tensor]) -> None:
        self.prefix_discriminator.set_valid_prefixes(valid_prefixes_by_depth)

    def generate(
        self,
        input_emb: Tensor,
        enc_mask_seq: Optional[Tensor] = None,
        max_length: Optional[int] = None,
        beam_sizes: Optional[Union[int, Sequence[int]]] = None,
        discriminator_pool_sizes: Optional[Union[int, Sequence[int]]] = None,
        temperature: float = 1.0,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        batch_size, seq_len, hidden_size = input_emb.shape
        if hidden_size != self.d_in_enc:
            raise ValueError(f"input_emb hidden size must be {self.d_in_enc}, got {hidden_size}")

        max_steps = 3 if max_length is None else max(1, min(int(max_length), 3))
        device = input_emb.device
        enc_mask_seq = self._prepare_enc_mask(enc_mask_seq, batch_size, seq_len, device)
        prefix_inputs = (input_emb, enc_mask_seq)
        bos_ids = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        h_step0 = self._decode_query_ids(input_emb, bos_ids, enc_mask_seq)
        logits0 = self._slice_step_logits(h_step0, step_idx=0)
        log_probs0 = F.log_softmax(logits0 / temperature, dim=-1)

        if beam_sizes is None or (isinstance(beam_sizes, Sequence) and len(beam_sizes) == 0):
            sid0 = torch.argmax(logits0, dim=-1)
            logprob_sum = log_probs0.gather(1, sid0.unsqueeze(1)).squeeze(1)
            if max_steps == 1:
                return sid0.unsqueeze(1), logprob_sum

            sid0_global = sid0.unsqueeze(1) + self.semantic_part_ranges[0][0]
            query_ids = torch.cat([bos_ids, sid0_global], dim=1)
            decoder_out = self._decode_query_ids(input_emb, query_ids, enc_mask_seq)
            logits1 = self._slice_step_logits(decoder_out, step_idx=1)
            log_probs1 = F.log_softmax(logits1 / temperature, dim=-1)
            sid1 = torch.argmax(logits1, dim=-1)
            logprob_sum = logprob_sum + log_probs1.gather(1, sid1.unsqueeze(1)).squeeze(1)
            if max_steps == 2:
                return torch.stack([sid0, sid1], dim=1), logprob_sum

            sid1_global = sid1.unsqueeze(1) + self.semantic_part_ranges[1][0]
            query_ids = torch.cat([bos_ids, sid0_global, sid1_global], dim=1)
            decoder_out = self._decode_query_ids(input_emb, query_ids, enc_mask_seq)
            logits2 = self._slice_step_logits(decoder_out, step_idx=2)
            log_probs2 = F.log_softmax(logits2 / temperature, dim=-1)
            sid2 = torch.argmax(logits2, dim=-1)
            logprob_sum = logprob_sum + log_probs2.gather(1, sid2.unsqueeze(1)).squeeze(1)
            return torch.stack([sid0, sid1, sid2], dim=1), logprob_sum

        beam_schedule = self._normalize_beam_sizes(beam_sizes)
        if discriminator_pool_sizes is None:
            pool_schedule = [int(k) for k in self.default_discriminator_pool_sizes]
        else:
            raw_pool_schedule = self._normalize_beam_sizes(discriminator_pool_sizes)
            pool_schedule = [max(pool_k, beam_k) for pool_k, beam_k in zip(raw_pool_schedule, beam_schedule)]

        pool0 = min(pool_schedule[0], log_probs0.size(-1))
        beam0 = min(beam_schedule[0], pool0)
        top_scores0, top_indices0 = torch.topk(log_probs0, k=pool0, dim=-1)
        pool_sids = top_indices0.unsqueeze(-1)
        aux_scores0 = self.prefix_discriminator.score_candidates(
            input_emb=input_emb,
            input_mask=enc_mask_seq,
            cand_sids=pool_sids,
            depth=1,
        )
        selected0 = torch.topk(aux_scores0, k=beam0, dim=-1).indices
        beam_ids = self._gather_beam_tensor(pool_sids, selected0)
        beam_scores = self._gather_beam_tensor(top_scores0, selected0)
        if max_steps == 1:
            return beam_ids, beam_scores

        current_k = beam_ids.size(1)
        prefix_emb_exp, enc_mask_exp = self._expand_prefix_inputs(prefix_inputs, current_k)
        for step_idx in range(1, max_steps):
            bos_beam = torch.zeros(batch_size, current_k, 1, device=device, dtype=torch.long)
            if step_idx == 1:
                sid0_global = beam_ids[:, :, 0:1] + self.semantic_part_ranges[0][0]
                query_ids = torch.cat([bos_beam, sid0_global], dim=-1)
                next_range = self.semantic_part_ranges[1]
            else:
                sid0_global = beam_ids[:, :, 0:1] + self.semantic_part_ranges[0][0]
                sid1_global = beam_ids[:, :, 1:2] + self.semantic_part_ranges[1][0]
                query_ids = torch.cat([bos_beam, sid0_global, sid1_global], dim=-1)
                next_range = self.semantic_part_ranges[2]

            decoder_out = self._decode_query_ids(
                prefix_emb_exp,
                query_ids.reshape(batch_size * current_k, query_ids.size(-1)),
                enc_mask_exp,
            )
            logits_all = self._slice_step_logits(decoder_out, step_idx=step_idx)
            part_vocab_size = next_range[1] - next_range[0]
            log_probs_part = F.log_softmax(logits_all / temperature, dim=-1).view(
                batch_size, current_k, part_vocab_size
            )
            total_scores = beam_scores[:, :, None] + log_probs_part
            pool_k = min(pool_schedule[step_idx], current_k * part_vocab_size)
            top_scores, top_indices = torch.topk(
                total_scores.view(batch_size, current_k * part_vocab_size),
                k=pool_k,
                dim=-1,
            )
            pool_beam_idx = torch.div(top_indices, part_vocab_size, rounding_mode="floor")
            pool_token_idx = top_indices % part_vocab_size
            pool_prev = torch.gather(
                beam_ids,
                dim=1,
                index=pool_beam_idx.unsqueeze(-1).expand(batch_size, pool_k, beam_ids.size(-1)),
            )
            pool_sids = torch.cat([pool_prev, pool_token_idx.unsqueeze(-1)], dim=-1)
            aux_scores = self.prefix_discriminator.score_candidates(
                input_emb=input_emb,
                input_mask=enc_mask_seq,
                cand_sids=pool_sids,
                depth=step_idx + 1,
            )
            next_k = min(beam_schedule[step_idx], pool_k)
            selected = torch.topk(aux_scores, k=next_k, dim=-1).indices
            beam_ids = self._gather_beam_tensor(pool_sids, selected)
            beam_scores = self._gather_beam_tensor(top_scores, selected)

            if step_idx < max_steps - 1:
                current_k = beam_ids.size(1)
                prefix_emb_exp, enc_mask_exp = self._expand_prefix_inputs(prefix_inputs, current_k)

        return beam_ids, beam_scores
