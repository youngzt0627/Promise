import argparse
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from generative_recommender_impl.amazon_data import (
    AmazonBatch,
    AmazonIdVocab,
    AmazonRandomEmbeddingAdapter,
    AmazonSidDataset,
)
DEFAULT_TRAIN_PATH = "./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
DEFAULT_TEST_PATH = "./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"


class EvalMetrics:
    def __init__(
        self,
        loss: float,
        sid0_acc: float,
        sid1_acc: float,
        sid2_acc: float,
        exact_match: float,
        greedy_exact_match: float,
    ) -> None:
        self.loss = loss
        self.sid0_acc = sid0_acc
        self.sid1_acc = sid1_acc
        self.sid2_acc = sid2_acc
        self.exact_match = exact_match
        self.greedy_exact_match = greedy_exact_match


def to_device(batch: AmazonBatch, device: torch.device) -> AmazonBatch:
    return AmazonBatch(
        user_ids=batch.user_ids.to(device),
        history_item_ids=batch.history_item_ids.to(device),
        history_title_ids=batch.history_title_ids.to(device),
        history_mask=batch.history_mask.to(device),
        target_sid=batch.target_sid.to(device),
    )


def build_model_and_data(
    train_path: str,
    test_path: str,
    hidden_size: int,
    sample_limit_train: Optional[int],
    sample_limit_test: Optional[int],
):
    vocab = AmazonIdVocab()
    vocab.build_from_csv([train_path])

    train_dataset = AmazonSidDataset(
        csv_path=train_path,
        vocab=vocab,
        max_history_len=None,
        sample_limit=sample_limit_train,
    )
    test_dataset = AmazonSidDataset(
        csv_path=test_path,
        vocab=vocab,
        max_history_len=train_dataset.max_history_len,
        sample_limit=sample_limit_test,
    )
    model = AmazonRandomEmbeddingAdapter(
        num_users=vocab.num_users,
        num_items=vocab.num_items,
        num_titles=vocab.num_titles,
        max_history_len=train_dataset.max_history_len,
        hidden_size=hidden_size,
    )
    model.set_valid_prefixes(train_dataset.valid_prefixes_by_depth)
    return model, train_dataset, test_dataset, vocab


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    max_train_batches: Optional[int] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch_idx, batch in enumerate(data_loader):
        if max_train_batches is not None and batch_idx >= max_train_batches:
            break
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            user_ids=batch.user_ids,
            history_item_ids=batch.history_item_ids,
            history_title_ids=batch.history_title_ids,
            history_mask=batch.history_mask,
            target_sid=batch.target_sid,
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(
    model: AmazonRandomEmbeddingAdapter,
    data_loader: DataLoader,
    device: torch.device,
    max_eval_batches: Optional[int] = None,
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_examples = 0
    sid_hits = torch.zeros(3, dtype=torch.float64)
    exact_hits = 0.0
    greedy_exact_hits = 0.0

    for batch_idx, batch in enumerate(data_loader):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break
        batch = to_device(batch, device)
        outputs = model(
            user_ids=batch.user_ids,
            history_item_ids=batch.history_item_ids,
            history_title_ids=batch.history_title_ids,
            history_mask=batch.history_mask,
            target_sid=batch.target_sid,
        )
        total_loss += float(outputs["loss"].item())
        total_batches += 1

        logits = outputs["ctr_logits"]
        pred0 = logits[:, 0, :].argmax(dim=-1)
        pred1 = logits[:, 1, :].argmax(dim=-1)
        pred2 = logits[:, 2, :].argmax(dim=-1)
        pred_local = torch.stack([pred0, pred1, pred2], dim=1)

        matches = pred_local.eq(batch.target_sid)
        sid_hits += matches.sum(dim=0).cpu().to(torch.float64)
        exact_hits += matches.all(dim=1).sum().item()

        greedy_ids, _ = model.generate(
            user_ids=batch.user_ids,
            history_item_ids=batch.history_item_ids,
            history_title_ids=batch.history_title_ids,
            history_mask=batch.history_mask,
            max_length=3,
            beam_sizes=None,
            temperature=1.0,
        )
        greedy_exact_hits += greedy_ids.eq(batch.target_sid).all(dim=1).sum().item()
        total_examples += batch.target_sid.size(0)

    denom = max(total_examples, 1)
    return EvalMetrics(
        loss=total_loss / max(total_batches, 1),
        sid0_acc=float(sid_hits[0].item() / denom),
        sid1_acc=float(sid_hits[1].item() / denom),
        sid2_acc=float(sid_hits[2].item() / denom),
        exact_match=exact_hits / denom,
        greedy_exact_match=greedy_exact_hits / denom,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train generative_recommender_impl on Amazon CSV data")
    parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit_train", type=int, default=0)
    parser.add_argument("--sample_limit_test", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_limit_train = args.sample_limit_train if args.sample_limit_train > 0 else None
    sample_limit_test = args.sample_limit_test if args.sample_limit_test > 0 else None
    max_train_batches = args.max_train_batches if args.max_train_batches > 0 else None
    max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

    model, train_dataset, test_dataset, vocab = build_model_and_data(
        train_path=args.train_path,
        test_path=args.test_path,
        hidden_size=args.hidden_size,
        sample_limit_train=sample_limit_train,
        sample_limit_test=sample_limit_test,
    )
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(
        {
            "train_path": args.train_path,
            "test_path": args.test_path,
            "num_train": len(train_dataset),
            "num_test": len(test_dataset),
            "num_users": vocab.num_users,
            "num_items": vocab.num_items,
            "num_titles": vocab.num_titles,
            "max_history_len": train_dataset.max_history_len,
            "device": str(device),
        }
    )

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_train_batches=max_train_batches,
        )
        metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            max_eval_batches=max_eval_batches,
        )
        payload: Dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
        }
        payload.update(
            {
                "loss": metrics.loss,
                "sid0_acc": metrics.sid0_acc,
                "sid1_acc": metrics.sid1_acc,
                "sid2_acc": metrics.sid2_acc,
                "exact_match": metrics.exact_match,
                "greedy_exact_match": metrics.greedy_exact_match,
            }
        )
        print(payload)


if __name__ == "__main__":
    main()
