import requests
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from threading import Thread

def call(prompt: str):
    start_t = datetime.now()

    requests.post(
        "http://localhost:8082/v1/motion",
        json={"text": prompt},
    )

    delta_t = (datetime.now() - start_t)

    use_time = delta_t.seconds +  delta_t.microseconds/ 1e6

    print(use_time)


if __name__ == "__main__":
    df = pd.read_csv("./server/test/introduce.csv")
    prompts = df.get("text_en").to_list()

    times = []
    # for prompt in tqdm(prompts[0:10]):
    threads = []
    for prompt in prompts[0:10]:
        thread = Thread(target=call, args=(prompt,))
        thread.start()
        threads.append(thread)

    for (i, thread) in enumerate(threads):
        # print(f"{i}: {thread.join()}")
        thread.join()

    