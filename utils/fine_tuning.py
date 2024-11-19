from datasets import Dataset
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from rapidfuzz import process, fuzz

with open("schema.json", "r") as f:
    schema = json.load(f)

with open("prompt.txt", "r") as file:
    json_prompt = file.read()

def formatting_prompts_func(examples, schema_str):
    inputs = examples["note_text"]
    responses = [
        {"HPI_Interval_Hx_start": hpi_start, "HPI_Interval_Hx_end": hpi_end, "A&P_start": ap_start, "A&P_end": ap_end}
        for hpi_start, hpi_end, ap_start, ap_end in zip(
            examples["HPI_Interval_Hx_start_str_gt"], 
            examples["HPI_Interval_Hx_end_str_gt"], 
            examples["A&P_start_str_gt"], 
            examples["A&P_end_str_gt"]
        )
    ]
    conversations = []
    for input_text, response_json in zip(inputs, responses):
        response_str = json.dumps(response_json)
        prompt = {"content": json_prompt.format(input_text, schema_str), "role": "user"}
        answer = {"content": response_str, "role": "assistant"}
        conversations.append([prompt, answer])
    
    return {"conversations": conversations}

def formatting_prompts_func_2(examples, tokenizer):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def generate_prompts(df, tokenizer):
    df_selected = df[["note_text", "HPI_Interval_Hx_start_str_gt", "HPI_Interval_Hx_end_str_gt", "A&P_start_str_gt", "A&P_end_str_gt"]].copy()
    df_selected = df_selected.fillna("")
    dataset_gen = Dataset.from_pandas(df_selected)
    schema_str = json.dumps(schema, indent=4)
    dataset_gen = dataset_gen.map(
        lambda examples: formatting_prompts_func(examples, schema_str),
        batched=True
    )
    dataset = dataset_gen.map(
        lambda examples: formatting_prompts_func_2(examples, tokenizer),
        batched=True
    )
    dataset = dataset.remove_columns(["note_text", "HPI_Interval_Hx_start_str_gt", "HPI_Interval_Hx_end_str_gt", "A&P_start_str_gt", "A&P_end_str_gt"])
    return dataset

def inference(df, tokenizer, model, output_path):
    dataset = generate_prompts(df, tokenizer)
    results = []
    json_parsing_failed = []  # To keep track of rows that failed JSON parsing

    for conv in tqdm(dataset['conversations'], desc="Processing conversations"):
        retry_count = 0
        success = False
        while retry_count < 3 and not success:
            try:
                # Tokenize and generate
                inputs = tokenizer.apply_chat_template(
                    [conv[0]],
                    tokenize=True,
                    add_generation_prompt=True,  # Must add for generation
                    return_tensors="pt",
                ).to("cuda")
                
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=128,
                    use_cache=True,
                    temperature=0.1,
                    min_p=0.1
                )
                
                # Decode generated text and parse JSON
                gen_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:])
                res_dict = json.loads(gen_text[0][:-10])  # Attempt to parse the JSON string
                
                # Append the result and mark as successful
                results.append(res_dict)
                json_parsing_failed.append(0)  # Success, mark as 0 for no parsing failure
                success = True  # Exit the retry loop after success
                
            except json.JSONDecodeError:
                retry_count += 1
                if retry_count == 3:
                    # If it fails after 3 attempts, append an empty dictionary and mark as failed
                    results.append({})
                    json_parsing_failed.append(1)  # Mark as 1 for parsing failure
                    print(f"Failed to parse JSON for conversation after {retry_count} attempts.")

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results)

    # Rename the columns to include the "_str_pred" suffix
    results_df.columns = ["HPI_Interval_Hx_start_str_pred", "HPI_Interval_Hx_end_str_pred", 
                        "A&P_start_str_pred", "A&P_end_str_pred"]

    # Add the JSON parsing failure tracking column to results_df
    results_df['json_parsing_failed'] = json_parsing_failed

    # Concatenate the new columns with the original df DataFrame
    df = pd.concat([df, results_df], axis=1)
    df.to_csv(output_path, index=False)

# Adjusted function to take 'note', 'start', and 'end' from the same DataFrame row with optional fuzzy matching
def extract_section_from_note_from_row(row, start_col, end_col, use_fuzzy: bool = False, fuzz_threshold: int = 80):
    """
    Extracts a section from a clinical note given the start and end strings from the same DataFrame row,
    with optional fuzzy matching.

    :param row: A row from a DataFrame containing 'note_text', 'start', and 'end' columns.
    :param use_fuzzy: Boolean flag to enable fuzzy matching.
    :param fuzz_threshold: The minimum similarity score for fuzzy matching (0-100).
    :return: The extracted section text, or None if the section cannot be found.
    """
    note = row['note_text']
    start_string = row[start_col]
    end_string = row[end_col]
    
    if pd.isna(start_string) or pd.isna(end_string):
        return np.nan, np.nan

    if use_fuzzy:
        # Perform fuzzy matching for start_string
        start_match = process.extractOne(start_string, note.splitlines(), scorer=fuzz.partial_ratio, score_cutoff=fuzz_threshold)
        if start_match:
            start_string = start_match[0]
        
        # Perform fuzzy matching for end_string
        end_match = process.extractOne(end_string, note.splitlines(), scorer=fuzz.partial_ratio, score_cutoff=fuzz_threshold)
        if end_match:
            end_string = end_match[0]

    if start_string not in note or end_string not in note:
        return np.nan, np.nan

    start_index = note.find(start_string)
    end_index = note.find(end_string, start_index) + len(end_string)

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return np.nan, np.nan

    return start_index, end_index

def compute_metrics(df, start_col, end_col, start_pred_col, end_pred_col):
    def calc_metrics(row):
        gt_start, gt_end = row[start_col], row[end_col]
        pred_start, pred_end = row[start_pred_col], row[end_pred_col]
        em = int(gt_start == pred_start and gt_end == pred_end)
        intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start) + 1)
        pred_len = pred_end - pred_start + 1
        gt_len = gt_end - gt_start + 1
        precision = intersection / pred_len if pred_len > 0 else 0
        recall = intersection / gt_len if gt_len > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return pd.Series([em, precision, recall, f1_score], index=['EM', 'Precision', 'Recall', 'F1_Score'])

    result_df = df[[start_col, end_col, start_pred_col, end_pred_col]].copy()
    result_df[['EM', 'Precision', 'Recall', 'F1_Score']] = df.apply(calc_metrics, axis=1)
    return result_df

def create_summary(df, label):
    summary = df[['EM', 'Precision', 'Recall', 'F1_Score']].mean().reset_index()
    summary.columns = ['Metric', label]
    return summary

def process_dataframe(df):
    # Apply strict and fuzzy extraction
    df['HPI_Interval_Hx_start_pred_strict'], df['HPI_Interval_Hx_end_pred_strict'] = zip(*df.apply(lambda row: extract_section_from_note_from_row(row, "HPI_Interval_Hx_start_str_pred", "HPI_Interval_Hx_end_str_pred", use_fuzzy=False, fuzz_threshold=70), axis=1))
    df['HPI_Interval_Hx_start_pred_fuzzy'], df['HPI_Interval_Hx_end_pred_fuzzy'] = zip(*df.apply(lambda row: extract_section_from_note_from_row(row, "HPI_Interval_Hx_start_str_pred", "HPI_Interval_Hx_end_str_pred", use_fuzzy=True, fuzz_threshold=70), axis=1))
    df['A&P_start_pred_strict'], df['A&P_end_pred_strict'] = zip(*df.apply(lambda row: extract_section_from_note_from_row(row, "A&P_start_str_pred", "A&P_end_str_pred", use_fuzzy=False, fuzz_threshold=70), axis=1))
    df['A&P_start_pred_fuzzy'], df['A&P_end_pred_fuzzy'] = zip(*df.apply(lambda row: extract_section_from_note_from_row(row, "A&P_start_str_pred", "A&P_end_str_pred", use_fuzzy=True, fuzz_threshold=70), axis=1))
    
    # Compute metrics for HPI_Interval_Hx section with strict and fuzzy matching
    hpi_strict_df = compute_metrics(df, 'HPI_Interval_Hx_start_gt', 'HPI_Interval_Hx_end_gt', 'HPI_Interval_Hx_start_pred_strict', 'HPI_Interval_Hx_end_pred_strict')
    hpi_fuzzy_df = compute_metrics(df, 'HPI_Interval_Hx_start_gt', 'HPI_Interval_Hx_end_gt', 'HPI_Interval_Hx_start_pred_fuzzy', 'HPI_Interval_Hx_end_pred_fuzzy')
    
    # Compute metrics for A&P section with strict and fuzzy matching
    ap_strict_df = compute_metrics(df, 'A&P_start_gt', 'A&P_end_gt', 'A&P_start_pred_strict', 'A&P_end_pred_strict')
    ap_fuzzy_df = compute_metrics(df, 'A&P_start_gt', 'A&P_end_gt', 'A&P_start_pred_fuzzy', 'A&P_end_pred_fuzzy')
    
    # Create summary tables for both sections and both matching methods
    hpi_strict_summary = create_summary(hpi_strict_df, 'HPI_Interval_Hx_Strict')
    hpi_fuzzy_summary = create_summary(hpi_fuzzy_df, 'HPI_Interval_Hx_Fuzzy')
    ap_strict_summary = create_summary(ap_strict_df, 'A&P_Strict')
    ap_fuzzy_summary = create_summary(ap_fuzzy_df, 'A&P_Fuzzy')
    
    # Combine summaries side by side
    summary_table = pd.concat([hpi_strict_summary.set_index('Metric'),
                               hpi_fuzzy_summary.set_index('Metric'),
                               ap_strict_summary.set_index('Metric'),
                               ap_fuzzy_summary.set_index('Metric')], axis=1).reset_index()
    
    return summary_table