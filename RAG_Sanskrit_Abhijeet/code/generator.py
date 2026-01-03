from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_answer(context, query, max_length=200):
    prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if not answer.strip():
        return "No clear answer could be generated from the retrieved context."

    return answer