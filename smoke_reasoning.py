"""Quick smoke test for the Muse-Spark endpoint and modal_reasoning."""
import os, json
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(
    base_url="https://a-p-shukla--ep-muse-spark-open-816b-a42b-server.us-west.modal.direct/v1",
    api_key=f"{os.getenv('MODAL_PROXY_TOKEN_ID')}.{os.getenv('MODAL_PROXY_TOKEN_SECRET')}",
)

print("=== Test 1: basic JSON response ===")
resp = client.chat.completions.create(
    model="meta-models/Muse-Spark-Open-816B-A42B",
    messages=[
        {"role": "system", "content": 'Reply with ONLY this exact JSON, nothing else: {"status": "ok", "value": 42}'},
        {"role": "user",   "content": "ping"},
    ],
    temperature=0.1,
    max_tokens=50,
    stream=False,
    extra_body={"reasoning_effort": "low"},
)
content = resp.choices[0].message.content
print(f"finish_reason : {resp.choices[0].finish_reason}")
print(f"raw content   : {repr(content)}")
print(f"usage         : {resp.usage}")

print()
print("=== Test 2: full RCA prompt (mini) ===")
from modal_reasoning import SYSTEM_PROMPT

resp2 = client.chat.completions.create(
    model="meta-models/Muse-Spark-Open-816B-A42B",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            "[SAM 2.1 successfully isolated the anomaly region shown in the image.]\n\n"
            "=== SANITIZED BACKEND LOGS ===\n"
            "2026-11-02T14:32:01Z ERROR TypeError: cart.items[0] is undefined at CheckoutButton.tsx:42\n\n"
            "=== GIT DIFF ===\n"
            "--- a/src/components/CheckoutButton.tsx\n"
            "+++ b/src/components/CheckoutButton.tsx\n"
            "@@ -11,1 +11,1 @@\n"
            "-  const label = cart.items[0].name;\n"
            "+  const label = cart.items[0]?.name;\n"
        )},
    ],
    temperature=0.1,
    max_tokens=512,
    stream=False,
    extra_body={"reasoning_effort": "medium"},
)
content2 = resp2.choices[0].message.content
print(f"finish_reason : {resp2.choices[0].finish_reason}")
print(f"content length: {len(content2 or '')}")
print(f"raw content   : {repr((content2 or '')[:600])}")
