import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Summarize the risks in the fiscal policy section.",
        }
    ]
)

print(completion.choices[0].message.content)
