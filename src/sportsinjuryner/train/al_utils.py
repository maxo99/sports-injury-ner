from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from sportsinjuryner.train.constants import INJURY_KEYWORDS
from sportsinjuryner.train.ner_utils import find_keyword_offsets

# Define label list (must match training)
LABEL_LIST = [
    "O",
    "B-PLAYER",
    "I-PLAYER",
    "B-INJURY",
    "I-INJURY",
    "B-STATUS",
    "I-STATUS",
    "B-TEAM",
    "I-TEAM",
]

id2label = dict(enumerate(LABEL_LIST))


def compute_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: torch.device,
) -> dict[str, Any]:
    """
    Computes confidence and conflict metrics for a given text.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=2)[0]  # (seq_len, num_labels)
    max_probs, _ = torch.max(probs, dim=1)  # (seq_len,)

    # 1. Confidence Score (Min of max probs)
    min_conf = torch.min(max_probs).item()

    # 2. Conflict Detection
    predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    pred_labels = [id2label[p] for p in predictions]

    # Check for keyword presence
    has_injury_kw = len(find_keyword_offsets(text, INJURY_KEYWORDS, "INJURY")) > 0

    # Check for model prediction
    has_injury_pred = any("INJURY" in l for l in pred_labels)

    conflict = has_injury_kw != has_injury_pred

    return {
        "confidence": float(min_conf),
        "conflict": conflict,
        "has_injury_kw": has_injury_kw,
        "has_injury_pred": has_injury_pred,
    }
