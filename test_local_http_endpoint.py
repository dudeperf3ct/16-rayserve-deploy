from pprint import pprint
import requests
import time

text = "i like you!"

for i in range(10):
    st = time.perf_counter()
    endpoint = f"http://127.0.0.1:8000/router?txt={text}"
    response = requests.get(endpoint)
    print("Time taken: {:.3f} sec for request {}".format(time.perf_counter() - st, i))
    pprint(response.json())
