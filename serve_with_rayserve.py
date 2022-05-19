import ray
from ray import serve

from sentiment.model import SentimentBertModel


# connect to Ray cluster
ray.init(address="auto", namespace="serve")
# start ray serve runtime
serve.start(detached=True)


def sentiment_classifier(text: str):
    classifier = SentimentBertModel()
    return classifier.predict(text)


# add the decorator @serve.deployment to the router function to turn the function into a Serve Deployment object.
@serve.deployment
# input : Starlette request object
def router(request):
    txt = request.query_params["txt"]
    return sentiment_classifier(txt)


# deploy the router deployment object to the ray serve runtime
router.deploy()
