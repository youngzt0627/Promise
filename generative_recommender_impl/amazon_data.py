import ast
import re
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .model import GenerativeRecommender


SID_PATTERN = re.compile(r"<a_(\d+)><b_(\d+)><c_(\d+)>")


def parse_python_list(raw_value: object) -> List[object]:
    if isinstance(raw_value, list):
        return raw_value
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return list(ast.literal_eval(text))


def parse_sid_string(sid_text: str) -> List[int]:
    match = SID_PATTERN.fullmatch(str(sid_text).strip())
    if match is None:
        raise ValueError(f"Invalid SID format: {sid_text}")
    return [int(match.group(1)), int(match.group(2)), int(match.group(3))]


class AmazonBatch:
    def __init__(
        self,
        user_ids: Tensor,
        history_item_ids: Tensor,
        history_title_ids: Tensor,
        history_mask: Tensor,
        target_sid: Tensor,
    ) -> None:
        self.user_ids = user_ids
        self.history_item_ids = history_item_ids
        self.history_title_ids = history_title_ids
        self.history_mask = history_mask
        self.target_sid = target_sid


class AmazonIdVocab:
    def __init__(self) -> None:
        self.user_to_idx: Dict[str, int] = {"<unk>": 0}
        self.item_to_idx: Dict[int, int] = {"<pad>": 0}  # type: ignore[dict-item]
        self.title_to_idx: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    def build_from_csv(self, csv_paths: Sequence[str]) -> None:
        for path in csv_paths:
            frame = pd.read_csv(path)
            for _, row in frame.iterrows():
                user_id = str(row["user_id"])
                if user_id not in self.user_to_idx:
                    self.user_to_idx[user_id] = len(self.user_to_idx)

                history_item_ids = parse_python_list(row["history_item_id"])
                target_item_id = int(row["item_id"])
                for item_id in history_item_ids + [target_item_id]:
                    item_id = int(item_id)
                    if item_id not in self.item_to_idx:
                        self.item_to_idx[item_id] = len(self.item_to_idx)

                history_titles = [str(x) for x in parse_python_list(row["history_item_title"])]
                target_title = str(row["item_title"])
                for title in history_titles + [target_title]:
                    if title not in self.title_to_idx:
                        self.title_to_idx[title] = len(self.title_to_idx)

    @property
    def num_users(self) -> int:
        return len(self.user_to_idx)

    @property
    def num_items(self) -> int:
        return len(self.item_to_idx)

    @property
    def num_titles(self) -> int:
        return len(self.title_to_idx)

    def user_index(self, user_id: str) -> int:
        return self.user_to_idx.get(str(user_id), 0)

    def item_index(self, item_id: int) -> int:
        return self.item_to_idx.get(int(item_id), 0)

    def title_index(self, title: str) -> int:
        return self.title_to_idx.get(str(title), 1)


class AmazonSidDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        vocab: AmazonIdVocab,
        max_history_len: Optional[int] = None,
        sample_limit: Optional[int] = None,
    ) -> None:
        frame = pd.read_csv(csv_path)
        if sample_limit is not None and sample_limit > 0:
            frame = frame.iloc[:sample_limit].copy()
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab

        if max_history_len is None:
            max_history_len = 1
            for value in self.frame["history_item_id"].tolist():
                max_history_len = max(max_history_len, len(parse_python_list(value)))
        self.max_history_len = int(max_history_len)
        self.valid_prefixes_by_depth = self._build_valid_prefixes()

    def __len__(self) -> int:
        return len(self.frame)

    def _build_valid_prefixes(self) -> List[Tensor]:
        prefix_sets = [set(), set(), set()]
        for sid_text in self.frame["item_sid"].tolist():
            sid = parse_sid_string(str(sid_text))
            for depth in range(1, 4):
                prefix_sets[depth - 1].add(tuple(sid[:depth]))

        valid_prefixes: List[Tensor] = []
        for depth, prefix_set in enumerate(prefix_sets, start=1):
            sorted_prefixes = sorted(prefix_set)
            valid_prefixes.append(torch.tensor(sorted_prefixes, dtype=torch.long).view(-1, depth))
        return valid_prefixes

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        history_item_ids = [int(x) for x in parse_python_list(row["history_item_id"])]
        history_titles = [str(x) for x in parse_python_list(row["history_item_title"])]
        history_item_ids = history_item_ids[-self.max_history_len :]
        history_titles = history_titles[-self.max_history_len :]
        history_local = [self.vocab.item_index(item_id) for item_id in history_item_ids]
        history_title_local = [self.vocab.title_index(title) for title in history_titles]
        target_sid = parse_sid_string(str(row["item_sid"]))

        return {
            "user_id": self.vocab.user_index(str(row["user_id"])),
            "history_item_ids": history_local,
            "history_title_ids": history_title_local,
            "target_sid": target_sid,
        }

    def collate_fn(self, batch: List[Dict[str, object]]) -> AmazonBatch:
        batch_size = len(batch)
        history_item_ids = torch.zeros(batch_size, self.max_history_len, dtype=torch.long)
        history_title_ids = torch.zeros(batch_size, self.max_history_len, dtype=torch.long)
        history_mask = torch.zeros(batch_size, self.max_history_len, dtype=torch.long)
        user_ids = torch.zeros(batch_size, dtype=torch.long)
        target_sid = torch.zeros(batch_size, 3, dtype=torch.long)

        for i, sample in enumerate(batch):
            user_ids[i] = int(sample["user_id"])  # type: ignore[arg-type]
            sample_history = list(sample["history_item_ids"])  # type: ignore[arg-type]
            sample_titles = list(sample["history_title_ids"])  # type: ignore[arg-type]
            seq_len = min(len(sample_history), len(sample_titles), self.max_history_len)
            if seq_len > 0:
                history_item_ids[i, :seq_len] = torch.tensor(sample_history[:seq_len], dtype=torch.long)
                history_title_ids[i, :seq_len] = torch.tensor(sample_titles[:seq_len], dtype=torch.long)
                history_mask[i, :seq_len] = 1
            target_sid[i] = torch.tensor(sample["target_sid"], dtype=torch.long)  # type: ignore[arg-type]

        return AmazonBatch(
            user_ids=user_ids,
            history_item_ids=history_item_ids,
            history_title_ids=history_title_ids,
            history_mask=history_mask,
            target_sid=target_sid,
        )


class QFormerCompressor(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_queries: int = 10,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.empty(num_queries, hidden_size))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.query_ln = nn.LayerNorm(hidden_size)
        self.kv_ln = nn.LayerNorm(hidden_size)
        self.out_ln = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, seq_emb: Tensor, seq_mask: Optional[Tensor] = None) -> Tensor:
        batch_size = seq_emb.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_ln(queries)
        kv = self.kv_ln(seq_emb)

        key_padding_mask = None
        if seq_mask is not None:
            key_padding_mask = seq_mask.eq(0)
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad] = False

        out, _ = self.cross_attn(
            query=queries,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.out_ln(out)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class AmazonRandomEmbeddingAdapter(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_titles: int,
        max_history_len: int,
        hidden_size: int = 512,
        base_model: Optional[GenerativeRecommender] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_history_len = max_history_len
        self.base_model = base_model or GenerativeRecommender(d_in_enc=hidden_size)

        self.user_embedding = nn.Embedding(num_users, hidden_size)
        self.item_embedding = nn.Embedding(num_items, hidden_size)
        self.title_embedding = nn.Embedding(num_titles, hidden_size)
        self.item_qformer = QFormerCompressor(hidden_size=hidden_size, num_queries=10, num_heads=8)
        self.title_qformer = QFormerCompressor(hidden_size=hidden_size, num_queries=10, num_heads=8)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.user_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.item_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.title_embedding.weight, std=0.02)

    def set_valid_prefixes(self, valid_prefixes_by_depth: Sequence[Tensor]) -> None:
        self.base_model.set_valid_prefixes(valid_prefixes_by_depth)

    def build_input_emb(
        self,
        user_ids: Tensor,
        history_item_ids: Tensor,
        history_title_ids: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        user_emb = self.user_embedding(user_ids).unsqueeze(1)
        item_emb = self.item_embedding(history_item_ids)
        title_emb = self.title_embedding(history_title_ids)
        item_memory = self.item_qformer(item_emb, history_mask)
        title_memory = self.title_qformer(title_emb, history_mask)
        return torch.cat([user_emb, item_memory, title_memory], dim=1)

    def build_input_mask(self, user_ids: Tensor, history_mask: Tensor) -> Tensor:
        batch_size = user_ids.size(0)
        del history_mask
        total_len = 1 + self.item_qformer.num_queries + self.title_qformer.num_queries
        return torch.ones(batch_size, total_len, device=user_ids.device, dtype=torch.long)

    def forward(
        self,
        user_ids: Tensor,
        history_item_ids: Tensor,
        history_title_ids: Tensor,
        history_mask: Tensor,
        target_sid: Tensor,
    ) -> Dict[str, Tensor]:
        input_emb = self.build_input_emb(user_ids, history_item_ids, history_title_ids, history_mask)
        input_mask = self.build_input_mask(user_ids, history_mask)
        return self.base_model(
            input_emb=input_emb,
            enc_mask_seq=input_mask,
            target_sid=target_sid,
        )

    @torch.no_grad()
    def generate(
        self,
        user_ids: Tensor,
        history_item_ids: Tensor,
        history_title_ids: Tensor,
        history_mask: Tensor,
        max_length: Optional[int] = None,
        beam_sizes: Optional[object] = None,
        discriminator_pool_sizes: Optional[object] = None,
        temperature: float = 1.0,
    ):
        input_emb = self.build_input_emb(user_ids, history_item_ids, history_title_ids, history_mask)
        input_mask = self.build_input_mask(user_ids, history_mask)
        return self.base_model.generate(
            input_emb=input_emb,
            enc_mask_seq=input_mask,
            max_length=max_length,
            beam_sizes=beam_sizes,
            discriminator_pool_sizes=discriminator_pool_sizes,
            temperature=temperature,
        )
