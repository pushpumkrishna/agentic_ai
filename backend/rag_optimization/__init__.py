import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
print(groq_api_key)

client = Groq(api_key=groq_api_key)


completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": ""}],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
