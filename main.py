from huggingface_hub import InferenceClient
import os

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token="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"
)
print(client.text_generation("Hello!", max_new_tokens=10))
