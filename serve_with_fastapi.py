import ray
from ray import serve

from fastapi import FastAPI
from fastapi import Query
from starlette.responses import JSONResponse

from sentiment.model import SentimentBertModel

app = FastAPI()
ray.init(address="auto", namespace="classifier")
serve.start(detached=True)


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class Classifier:
    def __init__(self) -> None:
        self.classifier = SentimentBertModel(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )

    @app.get("/test")
    def root(self):
        return "Sentiment Classifier (0 -> Negative and 1 -> Positive)"

    @app.get("/healthcheck", status_code=200)
    def healthcheck(self):
        return "dummy check! Classifier is all ready to go!"

    @app.post("/classify")
    async def predict_sentiment(self, input_text: str = Query(..., min_length=2)):
        out_dict = self.classifier.predict(input_text)
        return JSONResponse(out_dict)


Classifier.deploy()
