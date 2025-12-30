# Install and import necessary libraries for free LLM inference
# We'll use HuggingFace's transformers and a small open-source model Mistral-7B-Instruct-v0.2
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd


dataset = pd.read_csv("./processed_resumes.csv")  # Load your dataset containing resumes

# Load model and tokenizer (first time will download, may take time and disk space)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Create a text-generation pipeline
llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=16, device_map="auto")

# Define the prompt template
def make_prompt(resume_text):
    return (
        "You are an expert recruiter evaluating candidates for a job. "
        "Based solely on the resume text provided below, classify the candidate as ACCEPTED (1) or REJECTED (0) "
        "for a general professional role.\n\n"
        "Consider the following criteria while making your decision:\n"
        "- The candidate has a known academic degree (not 'unknown') and a relevant field of study.\n"
        "- At least 2 years of total experience (YOE).\n"
        "- Has held at least one prior job or internship.\n"
        "- Possesses at least one certification.\n"
        "- Has completed one or more projects.\n"
        "- Includes a GitHub or LinkedIn profile.\n"
        "- Demonstrates knowledge of more than one programming language.\n"
        "- Shows any leadership or management experience.\n"
        "- Has received awards, honors, or has volunteered.\n"
        "- Has published work or research (optional but valued).\n\n"
        "Make your decision based on how well the resume reflects these qualities.\n\n"
        f"Resume:\n{resume_text}\n\n"
        "Label (ACCEPTED / REJECTED):"
    )


# Select a random sample of 150 resumes to label
sampled_df = dataset.sample(n=150, random_state=42).copy()

# Function to get LLM label for a single resume
def get_llm_label(resume_text):
    prompt = make_prompt(resume_text[:2000])  # Truncate if needed for context length
    output = llm_pipe(prompt)[0]['generated_text']
    # Extract label from output (look for ACCEPTED or REJECTED)
    if "ACCEPTED" in output.upper():
        return 1
    elif "REJECTED" in output.upper():
        return 0
    else:
        return -1

# Apply LLM labeling to the sample
sampled_df['LLM_Label'] = sampled_df['Resume'].apply(get_llm_label)

# Save or inspect the labeled sample
sampled_df[['Resume', 'LLM_Label']].to_csv("llm_labeled_sample.csv", index=False)