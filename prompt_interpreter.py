
import json
import ollama

def interpret_prompt(user_prompt):
    response = ollama.chat(
        model="qwen3:8b",
        think=False,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the product ID and investigation date "
                    "from the user's request. "
                    "Return only JSON with two fields: "
                    "product_id and date. "
                    "Use YYYY-MM-DD for dates. "
                    "If a value is missing or ambiguous, return null. "
                    "Never invent missing information."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        format="json",
        options={"temperature": 0}
    )

    return json.loads(response["message"]["content"])


if __name__ == "__main__":
    prompt = input("Enter investigation request: ")
    result = interpret_prompt(prompt)
    print("Extracted:", result)
