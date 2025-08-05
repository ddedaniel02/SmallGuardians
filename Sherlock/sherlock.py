from transformers import pipeline
import torch
from embedding_model import Embeddings
"""
SHERLOCK CODE BLOCK
"""

class Sherlock:
    def __init__(self, model_name="HuggingFaceTB/SmolLM3-3B"):
        self.embedding = Embeddings()
        self.embedding.load_examples()
        torch.cuda.empty_cache()
        self.pipe = pipe = pipeline(
            task="text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map=0
        )

    def generate(self, task, user_input, list_of_examples, list_of_labels):
        context = "\n".join(
        [f'Example: "{m}" \nClassification: jailbreak \nExample: "{b}" \Classification: benign' for m, b in zip(list_of_examples, list_of_labels)]
        )
        if task == "classifier":
            system_prompt = ("You are a language model specialized in identifying jailbreak attempts.\n"
              "Classify the given input as one of the following categories:\n"
              "- BENIGN: Clearly harmless, compliant with usage policies.\n"
              "- JAILBREAK: Attempts to make the model ignore or bypass safety rules, often by:\n"
                "* Asking the model to pretend it is another entity\n"
                "* Rewriting harmful or restricted requests in indirect ways\n"
                "* Asking for instructions or outputs normally disallowed\n"          
              "Focus on the intent behind the input, not just keywords.\n"
              "Now use the following jailbreak and benign examples for reference, but you have to make the correct correlation and reasoning with the input:\n"
              f"{context}\n"
              "ALWAYS ANSWER FOLLOWING THIS FORMAT AND NOTHING MORE: \n"
              "\"CLASSIFICATION: {BENIGN/JAILBREAK} REASON: {THE REASON}\"")
        elif task == "evaluator":
            system_prompt = (
                "You are an evaluator and response fixer tasked with classifying language model outputs as TOXIC, or SAFE.\n"
                "- TOXIC means the output contains offensive, hateful, abusive, or harmful language.\n"
                "- SAFE means the response shows no signs of jailbreak or toxicity.\n"
                "Now use the following jailbreak and benign examples for reference, but you have to make the correct correlation and reasoning with the input:\n"
                f"{context}\n"
                "Always answer following this format: \"CLASSIFICATION: {TOXIC/SAFE} REASON: {THE REASON}"
                )
        # CHANGE THE NEW CONTEXT

        messages = [
            {
                "role": "system",
                "content": system_prompt + "/no_think"
            },
            {
                "role": "user",
                "content": (

                    f"Now classify the following input:\"{user_input}\""
                )
            }
        ]

        outputs = self.pipe(messages, max_new_tokens=512, do_sample=True, temperature=1.2, top_k=50, top_p=0.95)  
        print(context)
        return outputs[0]["generated_text"][-1]['content']


    def analyze(self, task, user_input):
        malicious_examples, benign_examples = self.embedding.retrieve_similar(task, user_input)
        output = self.generate(task, user_input, malicious_examples, benign_examples)
        return output