import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# 1. Load prompts
# --------------------------------------------------

input_file = "Dataset/prompts.csv"

df = pd.read_csv(input_file)

print("Prompts loaded successfully!")
print(df)


# --------------------------------------------------
# 2. Load pre-trained Hugging Face model
# --------------------------------------------------

model_name = "distilgpt2"

print("\nLoading Hugging Face model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 does not have a padding token by default
tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")


# --------------------------------------------------
# 3. Generate response
# --------------------------------------------------

def generate_response(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=400
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove the original prompt from the generated text
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]

    return generated_text.strip()


# --------------------------------------------------
# 4. Prompting techniques
# --------------------------------------------------

def zero_shot_prompt(problem):

    return f"""
Answer the following question clearly and accurately.

Question:
{problem}

Answer:
"""


def one_shot_prompt(problem):

    return f"""
Example:

Question:
What is evaporation?

Answer:
Evaporation is the process by which a liquid changes into a gas.

Now answer the following question:

Question:
{problem}

Answer:
"""


def few_shot_prompt(problem):

    return f"""
Use the examples below as guidance.

Example 1:
Question: What is photosynthesis?
Answer: Photosynthesis is the process by which plants use sunlight to make food.

Example 2:
Question: What is saving?
Answer: Saving means keeping money aside for future use.

Now answer:

Question:
{problem}

Answer:
"""


def role_prompt(problem):

    return f"""
You are an experienced and helpful teacher.

Explain the following question clearly and simply.

Question:
{problem}

Answer:
"""


def cot_prompt(problem):

    return f"""
Analyze the question carefully.

Identify the important points, give a brief explanation,
and then provide the final answer.

Question:
{problem}

Answer:
"""


# --------------------------------------------------
# 5. Generate responses
# --------------------------------------------------

results = []

print("\nGenerating responses...\n")

for _, row in df.iterrows():

    problem_id = row["Problem_ID"]
    category = row["Category"]
    problem = row["Problem"]

    print(f"Processing Problem {problem_id}: {category}")

    # Zero-shot
    zero_prompt = zero_shot_prompt(problem)
    zero_response = generate_response(zero_prompt)

    # One-shot
    one_prompt = one_shot_prompt(problem)
    one_response = generate_response(one_prompt)

    # Few-shot
    few_prompt = few_shot_prompt(problem)
    few_response = generate_response(few_prompt)

    # Role prompting
    role_prompt_text = role_prompt(problem)
    role_response = generate_response(role_prompt_text)

    # Chain-of-Thought style
    cot_prompt_text = cot_prompt(problem)
    cot_response = generate_response(cot_prompt_text)

    results.append({
        "Problem_ID": problem_id,
        "Category": category,
        "Problem": problem,
        "Zero_Shot_Response": zero_response,
        "One_Shot_Response": one_response,
        "Few_Shot_Response": few_response,
        "Role_Prompt_Response": role_response,
        "CoT_Response": cot_response
    })


# --------------------------------------------------
# 6. Save generated responses
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "Dataset/generated_responses.csv",
    index=False
)

print("\nGenerated responses saved successfully!")


# --------------------------------------------------
# 7. Prompt comparison
# --------------------------------------------------

techniques = {
    "Zero-Shot": "Zero_Shot_Response",
    "One-Shot": "One_Shot_Response",
    "Few-Shot": "Few_Shot_Response",
    "Role Prompting": "Role_Prompt_Response",
    "Chain-of-Thought": "CoT_Response"
}

comparison_rows = []

for technique, column in techniques.items():

    word_counts = results_df[column].fillna("").apply(
        lambda x: len(str(x).split())
    )

    comparison_rows.append({
        "Technique": technique,
        "Number_of_Responses": len(results_df),
        "Average_Word_Count": round(word_counts.mean(), 2)
    })


comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    "Dataset/prompt_comparison.csv",
    index=False
)

print("Prompt comparison saved successfully!")


# --------------------------------------------------
# 8. Evaluation results
# --------------------------------------------------

evaluation_rows = []

for _, row in results_df.iterrows():

    for technique, column in techniques.items():

        response = str(row[column])

        word_count = len(response.split())

        if word_count >= 20:
            completeness_score = 5
        elif word_count >= 10:
            completeness_score = 4
        elif word_count >= 5:
            completeness_score = 3
        elif word_count > 0:
            completeness_score = 2
        else:
            completeness_score = 0

        evaluation_rows.append({
            "Problem_ID": row["Problem_ID"],
            "Category": row["Category"],
            "Technique": technique,
            "Response_Length": word_count,
            "Completeness_Score": completeness_score
        })


evaluation_df = pd.DataFrame(evaluation_rows)

evaluation_df.to_csv(
    "Dataset/evaluation_results.csv",
    index=False
)

print("Evaluation results saved successfully!")


# --------------------------------------------------
# 9. Completion message
# --------------------------------------------------

print("\n========================================")
print("PROMPT ENGINEERING PROJECT COMPLETED")
print("========================================")

print("\nGenerated files:")

print("1. Dataset/generated_responses.csv")
print("2. Dataset/prompt_comparison.csv")
print("3. Dataset/evaluation_results.csv")

print("\nPrompting techniques used:")

print("1. Zero-Shot")
print("2. One-Shot")
print("3. Few-Shot")
print("4. Role Prompting")
print("5. Chain-of-Thought")