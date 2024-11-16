import os
import sys
import time
import json
import subprocess
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional
from rapidfuzz import process, fuzz
import openai
from pydantic import BaseModel
from utils.structs import IntervalHistoryOutput

class LLMWrapper:
    def __init__(self, prompt, llm_type='local', model_params=None, context_window=8192,
                 dataset_name='GI', output_dir='../outputs/', grammar_file=None, processor=None):
        """
        Initializes the LLMWrapper.

        :param prompt: The main prompt to use with the LLM.
        :param llm_type: 'local' or 'api' to indicate which LLM to use.
        :param model_params: Dictionary of parameters specific to the model.
        :param context_window: Context window size (-c parameter for local LLM).
        :param dataset_name: Name of the dataset (e.g., 'GI').
        :param output_dir: Directory to save outputs.
        :param grammar_file: Path to the grammar JSON file for local LLM.
        :param processor: An instance of LabelProcessor to reorganize outputs.
        """
        self.prompt = prompt
        self.llm_type = llm_type
        self.model_params = model_params or {}
        self.context_window = context_window
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.grammar_file = grammar_file
        self.processor = processor  # Include the processor
        self.llm_process = None  # Process for the local LLM server

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        if self.llm_type == 'local':
            self.start_local_llm_server()
        elif self.llm_type == 'api':
            # Initialize OpenAI client
            self.client = openai
            self.model_name = self.model_params.get('model_name')
            if not self.model_name:
                raise ValueError("model_name must be specified in model_params for API-based LLM.")
            # Set OpenAI API key if provided
            api_key = self.model_params.get('api_key')
            if api_key:
                self.client.api_key = api_key
            else:
                # Ensure that OPENAI_API_KEY environment variable is set
                if not os.environ.get('OPENAI_API_KEY'):
                    raise ValueError("OpenAI API key must be set in environment variable OPENAI_API_KEY or provided in model_params as 'api_key'.")
        else:
            raise ValueError("Invalid llm_type. Must be 'local' or 'api'.")

    def start_local_llm_server(self):
        """
        Starts the local LLM server using the specified model parameters.
        """
        model_path = self.model_params.get('model_path')
        ngl = self.model_params.get('ngl', 1000)
        temp = self.model_params.get('temp', 0)
        grammar_file = self.grammar_file

        if not model_path:
            raise ValueError("Model path must be specified in model_params for local LLM.")

        # Build the command to start the local LLM server
        command = [
            "../../llama.cpp/build/bin/llama-server",
            "-m", model_path,
            "-c", str(self.context_window),
            "-ngl", str(ngl),
            "--temp", str(temp)
        ]

        # Include grammar if specified
        if grammar_file:
            # Read and process the grammar file
            with open(grammar_file, 'r') as f:
                grammar_json = json.load(f)
            # Convert the grammar JSON to a compact, minimized string
            grammar_str = json.dumps(grammar_json, separators=(',', ':'))
            command.extend(["--json-schema", grammar_str])

        # Start the server as a subprocess
        self.llm_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for the server to start
        time.sleep(5)  # Adjust if needed

    def stop_local_llm_server(self):
        """
        Stops the local LLM server if it is running.
        """
        if self.llm_process:
            self.llm_process.terminate()
            self.llm_process.wait()
            self.llm_process = None

    def generate_llm_outputs(self, df):
        """
        Generates LLM outputs for each row in the DataFrame.

        :param df: DataFrame containing the data.
        :return: DataFrame with LLM outputs added.
        """

        if self.llm_type == 'local':
            hpi_begin = []
            hpi_end = []

            for note in tqdm(df['note_text'], desc="Processing Notes"):
                prompt = self.prompt + note
                try:
                    result = self._generate_local_response(prompt)
                    if result:
                        # Assuming result is a JSON string
                        result_dict = json.loads(result)
                        hpi_begin_text = result_dict.get('HPI_Interval_Hx_begin', None)
                        hpi_end_text = result_dict.get('HPI_Interval_Hx_end', None)
                        if hpi_begin_text:
                            begin_words = hpi_begin_text.split()
                            hpi_begin.append(' '.join(begin_words[:5]))
                        else:
                            hpi_begin.append(None)
                        if hpi_end_text:
                            end_words = hpi_end_text.split()
                            hpi_end.append(' '.join(end_words[:5]))
                        else:
                            hpi_end.append(None)
                    else:
                        hpi_begin.append(None)
                        hpi_end.append(None)
                except Exception as e:
                    print(f"Error: {e}")
                    hpi_begin.append(None)
                    hpi_end.append(None)

            df['start_pred_string'] = hpi_begin
            df['end_pred_string'] = hpi_end

            # Close the local LLM server after processing
            self.close()

        elif self.llm_type == 'api':
            df = self._process_text_data(df, self.prompt, 'note_text', ['start_pred_string', 'end_pred_string'])

        else:
            raise ValueError("Invalid llm_type. Must be 'local' or 'api'.")

        # **Include the reorganize_outputs step here**
        if self.processor:
            df = self.processor.reorganize_outputs(df, IntervalHistoryOutput)

        return df

    def _generate_local_response(self, prompt):
        """
        Generates a response from the local LLM server.

        :param prompt: The prompt to send.
        :return: The response content.
        """
        url = "http://localhost:8080/completion"
        payload = {
            "prompt": prompt,
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json().get('content', None)
        return result

    def _process_text_data(self, df, prompt, text_column, new_cols):
        """
        Process text data in the specified column of the DataFrame using an API call.

        Args:
        df (pd.DataFrame): DataFrame containing the text data.
        prompt (str): The prompt to use.
        text_column (str): Column name in DataFrame that contains the text data to process.
        new_cols (list): List of column names for storing the API responses.

        Returns:
        pd.DataFrame: DataFrame with the new columns containing API responses.
        """
        chunks = df[text_column].tolist()
        responses = []  # List to hold the responses

        for input_text in tqdm(chunks, desc="Processing text data"):
            try:
                # Call the API function for each prompt
                response = self._openai_chat_completion_response(prompt, input_text)
                responses.append(response)  # Append the response to the list
            except Exception as e:
                print(f"An error occurred with text '{input_text}': {e}")
                responses.append(None)  # Append None in case of an error

        # Parse the responses to extract 'start_string' and 'end_string'
        start_strings = []
        end_strings = []

        for resp in responses:
            if resp:
                try:
                    resp_dict = json.loads(resp)
                    start_strings.append(resp_dict.get('start_string'))
                    end_strings.append(resp_dict.get('end_string'))
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                    start_strings.append(None)
                    end_strings.append(None)
            else:
                start_strings.append(None)
                end_strings.append(None)

        df[new_cols[0]] = start_strings
        df[new_cols[1]] = end_strings

        return df

    def _openai_chat_completion_response(self, prompt, input_text):
        """
        Generates a response from the OpenAI API using the specified logic.

        :param prompt: The system prompt.
        :param input_text: The user input text.
        :return: The function arguments from the API response.
        """
        # Convert Pydantic model to function schema
        schema = [self.client.pydantic_function_tool(IntervalHistoryOutput)]

        # Remove 'strict' field if present
        schema = self.remove_strict_field(schema)

        # Extract function name
        function_name = self.extract_name_value(schema)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            functions=schema,
            function_call={"name": function_name},
        )

        return completion.choices[0].message.tool_calls[0].function.arguments

    def remove_strict_field(self, data):
        # Iterate through each dictionary in the list
        for item in data:
            # Check if 'strict' is a key in the 'function' dictionary
            if 'strict' in item.get('function', {}):
                # Remove the 'strict' field
                del item['function']['strict']
        return data

    def extract_name_value(self, data):
        # Access the 'name' field in the 'function' dictionary
        name_value = data[0].get('name')
        return name_value

    def compute_metrics(self, df):
        """
        Computes the metrics for the dataframe, including fuzzy matching.

        :param df: DataFrame with LLM outputs.
        :return: DataFrames with strict and fuzzy metrics.
        """
        # Add fuzzy matching columns
        df['start_pred_strict'], df['end_pred_strict'] = zip(*df.apply(
            lambda row: self._extract_section_from_note_from_row(row, use_fuzzy=False), axis=1))
        df['start_pred_fuzzy'], df['end_pred_fuzzy'] = zip(*df.apply(
            lambda row: self._extract_section_from_note_from_row(row, use_fuzzy=True), axis=1))

        # Compute metrics for both strict and fuzzy
        strict_df = self._compute_metrics_for_df(df, 'start', 'end', 'start_pred_strict', 'end_pred_strict')
        fuzzy_df = self._compute_metrics_for_df(df, 'start', 'end', 'start_pred_fuzzy', 'end_pred_fuzzy')

        return strict_df, fuzzy_df

    def _extract_section_from_note_from_row(self, row, use_fuzzy=False, fuzz_threshold=80):
        note = row['note_text']
        start_string = row['start_pred_string']
        end_string = row['end_pred_string']

        if pd.isna(start_string) or pd.isna(end_string):
            return np.nan, np.nan

        if use_fuzzy:
            # Perform fuzzy matching for start_string
            start_match = process.extractOne(start_string, [note], scorer=fuzz.partial_ratio,
                                             score_cutoff=fuzz_threshold)
            if start_match:
                start_string = start_match[0]

            # Perform fuzzy matching for end_string
            end_match = process.extractOne(end_string, [note], scorer=fuzz.partial_ratio,
                                           score_cutoff=fuzz_threshold)
            if end_match:
                end_string = end_match[0]

        if start_string not in note or end_string not in note:
            return np.nan, np.nan

        start_index = note.find(start_string)
        end_index = note.find(end_string, start_index) + len(end_string)

        if start_index == -1 or end_index == -1 or start_index >= end_index:
            return np.nan, np.nan

        return start_index, end_index

    def _compute_metrics_for_df(self, df, start_col, end_col, start_pred_col, end_pred_col):
        def calc_metrics(row):
            gt_start, gt_end = row[start_col], row[end_col]
            pred_start, pred_end = row[start_pred_col], row[end_pred_col]
            if pd.isna(pred_start) or pd.isna(pred_end):
                return pd.Series([0, 0, 0, 0], index=['EM', 'Precision', 'Recall', 'F1_Score'])
            em = int(gt_start == pred_start and gt_end == pred_end)
            intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start) + 1)
            pred_len = pred_end - pred_start + 1
            gt_len = gt_end - gt_start + 1
            precision = intersection / pred_len if pred_len > 0 else 0
            recall = intersection / gt_len if gt_len > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return pd.Series([em, precision, recall, f1_score], index=['EM', 'Precision', 'Recall', 'F1_Score'])

        result_df = df[[start_col, end_col, start_pred_col, end_pred_col]].copy()
        metrics_df = df.apply(calc_metrics, axis=1)
        result_df = pd.concat([result_df, metrics_df], axis=1)
        return result_df

    def create_summary(self, df, label):
        summary = df[['EM', 'Precision', 'Recall', 'F1_Score']].mean().reset_index()
        summary.columns = ['Metric', label]
        return summary

    def save_results(self, df, strict_df, fuzzy_df):
        """
        Saves the outputs and summaries.

        :param df: DataFrame with LLM outputs.
        :param strict_df: DataFrame with strict metrics.
        :param fuzzy_df: DataFrame with fuzzy metrics.
        """
        # Save the LLM outputs
        output_path = os.path.join(self.output_dir, f"{self.dataset_name}_{self.llm_type}_outputs.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved LLM outputs to {output_path}")

        # Compute summaries
        strict_summary = self.create_summary(strict_df, 'Strict')
        fuzzy_summary = self.create_summary(fuzzy_df, 'Fuzzy')

        # Compute non-zero ratios
        non_zero_strict = strict_df[(strict_df['Precision'] != 0) & (strict_df['Recall'] != 0) & (strict_df['F1_Score'] != 0)]
        non_zero_fuzzy = fuzzy_df[(fuzzy_df['Precision'] != 0) & (fuzzy_df['Recall'] != 0) & (fuzzy_df['F1_Score'] != 0)]
        non_zero_strict_ratio = len(non_zero_strict) / len(df)
        non_zero_fuzzy_ratio = len(non_zero_fuzzy) / len(df)

        # Merge summaries
        summary_table = pd.merge(strict_summary, fuzzy_summary, on='Metric')

        # Add additional info
        summary_row = {
            'prompt': self.prompt,
            'llm_type': self.llm_type,
            'context_window': self.context_window,
            'dataset': self.dataset_name,
            'non_zero_strict_ratio': non_zero_strict_ratio,
            'non_zero_fuzzy_ratio': non_zero_fuzzy_ratio,
        }

        for idx, row in summary_table.iterrows():
            metric = row['Metric']
            summary_row[f'strict_{metric}'] = row['Strict']
            summary_row[f'fuzzy_{metric}'] = row['Fuzzy']

        # Load existing global CSV or create new
        global_csv_path = os.path.join(self.output_dir, 'global_summary.csv')
        if os.path.exists(global_csv_path):
            global_summary_df = pd.read_csv(global_csv_path)
        else:
            global_summary_df = pd.DataFrame()

        # Append new summary
        global_summary_df = global_summary_df.append(summary_row, ignore_index=True)

        # Save global summary
        global_summary_df.to_csv(global_csv_path, index=False)
        print(f"Updated global summary saved to {global_csv_path}")

    def close(self):
        """
        Cleans up resources, stops the local LLM server if needed.
        """
        self.stop_local_llm_server()