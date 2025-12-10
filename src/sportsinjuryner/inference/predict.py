import argparse
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from sportsinjuryner.config import setup_logging

logger = setup_logging(__name__)

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
label2id = {label: i for i, label in enumerate(LABEL_LIST)}


class InjuryNER:
    def __init__(self, model_path: str = "sports-injury-ner-model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {model_path} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                num_labels=len(LABEL_LIST),
                id2label=id2label,
                label2id=label2id,
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, text: str) -> list[dict[str, Any]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = []
        current_entity = None

        for idx, (_, pred_id, (start, end)) in enumerate(
            zip(tokens, predictions, offset_mapping, strict=False)
        ):
            # Skip special tokens
            if start == end:
                continue

            label = LABEL_LIST[pred_id]

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "entity": label[2:],
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "score": float(torch.softmax(logits[0][idx], dim=0)[pred_id]),
                }
            elif label.startswith("I-") and current_entity:
                if label[2:] == current_entity["entity"]:
                    # Append to current entity
                    # Handle subword tokens (usually start with ## or are contiguous)
                    # We just update the end position and text
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"] : end]
                    # Average score? Or keep min/max? Let's keep min for robustness
                    current_score = float(torch.softmax(logits[0][idx], dim=0)[pred_id])
                    current_entity["score"] = min(
                        current_entity["score"], current_score
                    )
                else:
                    # Mismatch I-tag, treat as new or ignore?
                    # Standard practice: end previous, start new if B- missing?
                    # For simplicity, we just end the previous one.
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Sports Injury NER model"
    )
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument(
        "--model", default="sports-injury-ner-model", help="Path to trained model"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    args = parser.parse_args()

    try:
        ner = InjuryNER(args.model)
    except Exception:
        logger.error("Could not initialize model. Ensure you have trained it first.")
        return

    if args.interactive or not args.text:
        print("Interactive Mode. Type 'q' to quit.")
        while True:
            text = input("\nEnter text: ")
            if text.lower() in ["q", "quit", "exit"]:
                break

            results = ner.predict(text)
            print("\nEntities found:")
            for ent in results:
                print(f"  {ent['entity']:<10} | {ent['text']:<20} | {ent['score']:.2f}")
    else:
        results = ner.predict(args.text)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import json

    main()
