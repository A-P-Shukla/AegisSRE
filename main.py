import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://a-p-shukla--ep-muse-spark-open-816b-a42b-server.us-west.modal.direct/v1",
    api_key=f"{os.getenv('MODAL_PROXY_TOKEN_ID')}.{os.getenv('MODAL_PROXY_TOKEN_SECRET')}",
)

completion = client.chat.completions.create(
    model="meta-models/Muse-Spark-Open-816B-A42B",
    messages=[
        {
            "role": "system",
            "content": "You are a concise technical assistant.",
        },
        {
            "role": "user",
            "content": "Explain why low latency matters for LLM endpoints in three bullets.",
        },
    ],
    temperature=0.3,
    max_tokens=2048,
    top_p=0.95,
    stream=False,
    extra_body={"reasoning_effort": "medium"},
)
print(completion.choices[0].message.content)