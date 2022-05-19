from collections import defaultdict

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


class SentimentBertModel:
    def __init__(
        self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running inference on {self.device}")
        self.model_name = model_name
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        logger.info(
            f"Using {self.model_name} finetuned model for sentiment classification"
        )

    def predict(self, text: str) -> dict:
        logger.info(f"Input text: {text}")
        inputs = self.tokenizer(text, return_tensors="pt")
        input_id = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        logger.info(f"Input ids: {input_id} Shape: {input_id.shape}")
        logger.info(
            f"Input Attention mask: {attention_mask} Shape: {attention_mask.shape}"
        )
        with torch.no_grad():
            outputs = self.model(input_id, attention_mask)
            probs = F.softmax(outputs.logits, dim=1).numpy()[0]
        logger.info(f"Logits: {outputs.logits.numpy()[0]}")
        logger.info(f"Probabilities: {probs}")
        d = self.create_dict(text, probs)
        return d

    def create_dict(self, text: str, probs: list) -> dict:
        d = defaultdict()
        d["input_text"] = text
        d["pos_label"] = "positive"
        d["pos_score"] = float(probs[1])
        d["neg_label"] = "negative"
        d["neg_score"] = float(probs[0])
        return d


if __name__ == "__main__":
    classifier = SentimentBertModel()
    print(classifier.predict("i like you!"))
    print(classifier.predict("i hate you!"))
