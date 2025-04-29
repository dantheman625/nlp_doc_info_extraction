from typing import List, Dict, Tuple

class NEREvaluator:
    """
    NER evaluator that takes ground truth & predictions directly.

    Args:
      text: the document string (optional, for reference)
      gt_entities: List of dicts from your GT file, each with:
           {
             "type": str,
             "mentions": List[str]
           }
      predicted_entities: List of dicts from your model, each:
           {
             "type": str,
             "text": str
           }
    """

    def __init__(
        self,
        text: str,
        gt_entities: List[Dict[str, any]],
        predicted_entities: List[Dict[str, str]],
    ):
        self.text = text

        # flatten the GT into (mention_text, type)
        self.gt_mentions: List[Tuple[str, str]] = []
        for ent in gt_entities:
            etype = ent["type"]
            for m in ent.get("mentions", []):
                self.gt_mentions.append((m, etype))

        # flatten predictions
        self.pred_mentions: List[Tuple[str, str]] = [
            (ent["text"], ent["type"]) for ent in predicted_entities
        ]

    def evaluate(self) -> Dict[str, float]:
        """
        Compute precision, recall, F1 on this single doc.
        Matching is exact on (text, type) pairs, with one-to-one removal.

        Returns:
          {"precision": P, "recall": R, "f1": F}
        """

        print("check mention sets")
        print(self.gt_mentions)
        print(self.pred_mentions)
        # copy GT for removal as we match
        remaining = self.gt_mentions.copy()
        tp = 0
        for m in self.pred_mentions:
            if m in remaining:
                tp += 1
                remaining.remove(m)

        fp = len(self.pred_mentions) - tp
        fn = len(self.gt_mentions) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if precision + recall > 0 else 0.0)

        return {"precision": precision, "recall": recall, "f1": f1}
