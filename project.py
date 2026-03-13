##Install dependancies
import os, re, torch, gc, matplotlib.pyplot as plt, numpy as np, collections, evaluate
## Copy and paste this for Colab, use as your first cell
"""
%%capture
import os, re, torch
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth  # Do this in local & cloud setups
else:
    import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
"""
## Other wise run this block in terminal and install
"""
pip install unsloth
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
pip install datasets
pip install evaluate
pip install rouge_score
"""

##Prep the data sets
from datasets import load_dataset, concatenate_datasets

pubmedqa_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
medquad_dataset = load_dataset("lavita/MedQuAD", split="train")
legalqaeval_dataset = load_dataset("isaacus/LegalQAEval", split= ["val", "test"])
# cap_dataset = load_dataset("free-law/Caselaw_Access_Project", split="train")
def prepare_pubmedqa_columns(example):
    return {

        #merge final_decision and long)answer into a single column
        "output": f"{example['final_decision'].capitalize()}. {example['long_answer']}"
    }

def prepare_legalqaeval_columns(example):
    # Safely extract the fields, defaulting to empty strings
    ctx = example.get("text") or example.get("context") or ""
    q = example.get("question") or ""
    ans = example.get("answer") or example.get("answers") or ""

    # Handle the nested dictionary or list formats found in LegalQA
    if isinstance(ans, dict) and "text" in ans:
        ans = ans["text"]
    if isinstance(ans, list):
        ans = ans[0] if ans else ""

    return {
        "question": str(q),
        "context": str(ctx),
        "answer": str(ans).strip()
    }

pubmedqa_dataset = pubmedqa_dataset.map(prepare_pubmedqa_columns)
legalqaeval_dataset = concatenate_datasets(legalqaeval_dataset)
legalqaeval_dataset = legalqaeval_dataset.map(prepare_legalqaeval_columns)
# Filter out any rows where the answer/output is missing or literally says "None"
pubmedqa_dataset = pubmedqa_dataset.filter(lambda x: x["output"] is not None and str(x["output"]).strip() != "None" and str(x["output"]).strip() != "")
medquad_dataset = medquad_dataset.filter(lambda x: x["answer"] is not None and str(x["answer"]).strip() != "None" and str(x["answer"]).strip() != "")
legalqaeval_dataset = legalqaeval_dataset.filter(lambda x: x["answer"] is not None and str(x["answer"]).strip() not in ["None", ""])

#Merge columns into 2 columns, question and answer
from unsloth import to_sharegpt

def merge_prompt(dataset, merger, output):
  return to_sharegpt(
    dataset,
    merged_prompt = merger,
    output_column_name = output,
    conversation_extension = 0,  # Select more to handle longer conversations
)
pubmedqa_dataset = merge_prompt(pubmedqa_dataset, "[[ Question: {question}, Context: {context}]]", "output")
medquad_dataset = merge_prompt(medquad_dataset,"[[Question: {question}; Question Focus and Type: {question_focus}, {question_type}; Context URL: {document_url}]]" , "answer" )
legalqaeval_dataset = merge_prompt(legalqaeval_dataset, "[[ Question: {question}, Context: {context} ]]", "answer")

##Standardize the dataset with sharegpt
from unsloth import standardize_sharegpt

pubmedqa_dataset = standardize_sharegpt(pubmedqa_dataset)
medquad_dataset = standardize_sharegpt(medquad_dataset)
legalqaeval_dataset = standardize_sharegpt(legalqaeval_dataset)

##Apply chat template
chat_template = """Below is a question. Write a response that correctly answers the question.

### Question:
{INPUT}

### Response:
{OUTPUT}"""

from unsloth import apply_chat_template

def apply_chat_template_dynamic(dataset, tokenizer, chat_template):
  return apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)

## Functions to evaluate F1 and ROUGE score
from tqdm import tqdm


def compute_token_f1(prediction, truth):
    """Calculates Token-level F1 score between two strings."""
    pred_tokens = prediction.lower().split()
    truth_tokens = truth.lower().split()

    # If both are empty, perfect match
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    # If one is empty and the other isn't, 0 match
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    # Count common words
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def run_f1_evaluation(model, tokenizer, dataset, is_unsloth):
    """Generates answers for a dataset and calculates average F1 using a custom template."""

    if is_unsloth:
        FastLanguageModel.for_inference(model)

    total_f1 = 0.0

    print("Generating answers for evaluation...")
    for item in tqdm(dataset):
        # Extract the question and answer from your dataset.
        # (Adjust these keys if your dataset columns are named differently)
        user_question = item["conversations"][0]["content"]
        ground_truth = item["conversations"][1]["content"]

        # 1. Format the prompt EXACTLY as your custom template dictates,
        # but leave the response area empty!
        inference_prompt = f"""Below is a question. Write a a response that correctly answers the question.

### Question:
{user_question}

### Response:
"""

        # 2. Tokenize the raw string directly (no need for apply_chat_template here!)
        inputs = tokenizer(inference_prompt, return_tensors="pt").to("cuda")

        # 3. Generate the answer
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # 4. Decode the output and slice off the prompt tokens to get just the new words
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 5. Calculate F1 for this specific question
        f1 = compute_token_f1(generated_text, ground_truth)
        total_f1 += f1

    avg_f1 = total_f1 / len(dataset)
    return avg_f1

def run_rouge_evaluation(model, tokenizer, dataset_name, num_samples=50):
    """
    Evaluates a model on summarization datasets (billsum or scientific_papers)
    using ROUGE metrics.
    """
    # 1. Load Metric and Data
    rouge = evaluate.load("rouge")

    if dataset_name == "billsum":
        ds = load_dataset("billsum", split=f"test[:{num_samples}]")
        text_key, target_key = "text", "summary"
        instr = "Summarize the following bill."
    elif dataset_name == "scientific_papers":
        ds = load_dataset("ccdv/pubmed-summarization", split=f"test[:{num_samples}]")
        text_key, target_key = "article", "abstract"
        instr = "Summarize the following paper."
    else:
        raise ValueError("Unsupported dataset. Choose 'billsum' or 'scientific_papers'.")

    # 2. Set model to inference mode
    FastLanguageModel.for_inference(model)

    preds = []
    refs = [item[target_key] for item in ds]

    print(f"Generating {num_samples} summaries for {dataset_name}...")

    for item in tqdm(ds):
        # 3. Format the Prompt
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{item[text_key]}\n\n### Response:\n"

        # 4. Tokenize and Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=256, # Summaries usually need more than 100 tokens
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # 5. Decode and Extract Response
        # Slicing inputs.input_ids.shape[1] ensures we don't include the prompt in the score
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        preds.append(generated_text.strip())

    # 6. Compute ROUGE
    results = rouge.compute(predictions=preds, references=refs)
    return results