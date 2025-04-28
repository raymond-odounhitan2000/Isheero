import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_type = "azure"
openai.api_base = os.getenv("endpoints")
openai.api_key = os.getenv("API_KEY")
openai.api_version = os.getenv("API_VERSION")

def annotate_description(prompt):
    response = openai.ChatCompletion.create(
        engine=os.getenv("DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "Tu es un expert en vision par ordinateur. Décris précisément des images africaines."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    prompt = "Décris une image montrant un homme vendant des fruits dans un village africain."
    result = annotate_description(prompt)
    print(result)
