import pandas as pd
import json


class LabelProcessor:
    """
    A class for processing labels, comparing them, and calculating accuracy 
    between two dataframes (k_df and j_df).
    """

    def __init__(self, k_df, j_df):
        """
        Initialize the LabelProcessor with two dataframes: k_df and j_df.
        These dataframes contain labels for comparison.
        """
        self.k_df = k_df  # DataFrame containing k_label data
        self.j_df = j_df  # DataFrame containing j_label data
        self.result_df = None  # To store the result after processing

    @staticmethod
    def compare_labels_only(row1, row2):
        """
        Compare the labels in two rows based only on the 'labels' field.
        
        Args:
        - row1: The first row (k_label).
        - row2: The second row (j_label).
        
        Returns:
        - True if labels are identical, otherwise False.
        """
        if pd.isna(row1) and pd.isna(row2):
            return True
        if pd.isna(row1) or pd.isna(row2):
            return False

        # Extract 'labels' field from both rows
        def extract_labels(data):
            return [item.get('labels', []) for item in data if isinstance(item, dict)]

        labels_row1 = extract_labels(row1 if isinstance(row1, list) else [row1])
        labels_row2 = extract_labels(row2 if isinstance(row2, list) else [row2])
        
        return sorted(labels_row1) == sorted(labels_row2)

    @staticmethod
    def process_labels(json_string):
        """
        Convert a JSON string of labels into a list of processed label sections.
        
        Args:
        - json_string: A JSON-formatted string containing label data.
        
        Returns:
        - A list of sections, each containing 'start', 'end', and 'type' (label).
        """
        if not isinstance(json_string, str):
            return []
        try:
            parsed_value = json.loads(json_string)
        except (json.JSONDecodeError, TypeError):
            return []
        
        sections = []
        for item in parsed_value:
            labels = item["labels"]  # List of labels
            start = item["start"]    # Start position of the label
            end = item["end"]        # End position of the label
            for label in labels:
                sections.append({"start": start, "end": end, "type": label})
        return sections

    @staticmethod
    def merge_spans(spans):
        """
        Merge overlapping or contiguous spans.

        Args:
        - spans: List of spans, each with 'start' and 'end'.

        Returns:
        - A list of merged spans.
        """
        if not spans:
            return []
        
        # Sort spans by start
        sorted_spans = sorted(spans, key=lambda x: x['start'])
        merged = [sorted_spans[0]]
        
        for current in sorted_spans[1:]:
            last = merged[-1]
            if current['start'] <= last['end']:  # Overlapping or contiguous
                merged[-1]['end'] = max(last['end'], current['end'])
            else:
                merged.append(current)
        
        return merged

    @staticmethod
    def calculate_intersection(merged_L1, merged_L2):
        """
        Calculate the total intersection length between two lists of merged spans.

        Args:
        - merged_L1: Merged spans from L1.
        - merged_L2: Merged spans from L2.

        Returns:
        - Total intersection length.
        """
        i, j = 0, 0
        intersection = 0
        
        while i < len(merged_L1) and j < len(merged_L2):
            a = merged_L1[i]
            b = merged_L2[j]
            
            # Find overlap between a and b
            start = max(a['start'], b['start'])
            end = min(a['end'], b['end'])
            
            if start < end:
                intersection += end - start
            
            # Move to the next span
            if a['end'] < b['end']:
                i += 1
            else:
                j += 1
        
        return intersection

    @staticmethod
    def calculate_union(merged_L1, merged_L2):
        """
        Calculate the total union length between two lists of merged spans.

        Args:
        - merged_L1: Merged spans from L1.
        - merged_L2: Merged spans from L2.

        Returns:
        - Total union length.
        """
        all_spans = merged_L1 + merged_L2
        if not all_spans:
            return 0
        
        # Merge all spans to get the union
        merged_union = [all_spans[0]]
        
        for current in sorted(all_spans, key=lambda x: x['start'])[1:]:
            last = merged_union[-1]
            if current['start'] <= last['end']:
                merged_union[-1]['end'] = max(last['end'], current['end'])
            else:
                merged_union.append(current)
        
        union_length = sum(span['end'] - span['start'] for span in merged_union)
        return union_length

    def evaluate_labels(self, L1, L2, all_possible_types=["HPI_Interval_Hx", "A&P"]):
        """
        Compare labels between two researchers to calculate the overlap ratio using Intersection over Union (IoU).

        Args:
        - L1: A list of spans from researcher 1, each with start, end, and type.
        - L2: A list of spans from researcher 2.
        - all_possible_types: (Optional) A list of all possible types.

        Returns:
        - A dictionary mapping labels to their IoU overlap ratio.
        """
        overlaps = dict()

        for label_type in all_possible_types:
            spans_L1 = [s for s in L1 if s['type'] == label_type]
            spans_L2 = [s for s in L2 if s['type'] == label_type]

            # If both researchers have no annotations for this type, set overlap to 1
            if not spans_L1 and not spans_L2:
                overlaps[label_type] = 1.0
                continue

            # Merge spans within each list
            merged_L1 = self.merge_spans(spans_L1)
            merged_L2 = self.merge_spans(spans_L2)

            # Calculate intersection and union
            intersection = self.calculate_intersection(merged_L1, merged_L2)
            union = self.calculate_union(merged_L1, merged_L2)

            # Handle edge case where union is zero
            if union == 0:
                overlap_ratio = 1.0
            else:
                overlap_ratio = intersection / union

            overlaps[label_type] = overlap_ratio

        return overlaps

    def calculate_row_accuracy(self, row):
        """
        Calculate accuracy for a single row by processing k_label and j_label.

        Args:
        - row: A DataFrame row containing 'k_label' and 'j_label'.

        Returns:
        - A dictionary of overlaps between the two labels.
        """
        L1 = self.process_labels(row['k_label'])
        L2 = self.process_labels(row['j_label'])
        return self.evaluate_labels(L1, L2)

    def process(self):
        """
        Process the entire dataset to compare labels and calculate accuracies.

        Returns:
        - A DataFrame with accuracy results.
        """
        self.result_df = pd.DataFrame()
        self.result_df['k_label'] = self.k_df['label'].copy()
        self.result_df['j_label'] = self.j_df['label'].copy()

        # Compare the labels between j_df and k_df
        self.result_df['label_match'] = self.j_df['label'].combine(self.k_df['label'], self.compare_labels_only)

        # Remove rows where the labels don't match
        false_index = self.result_df[self.result_df['label_match'] == False].index
        self.result_df = self.result_df.drop(false_index)

        # Calculate accuracy for each row
        accuracy_df = self.result_df.apply(self.calculate_row_accuracy, axis=1, result_type="expand")
        self.result_df = pd.concat([self.result_df, accuracy_df], axis=1)

        return self.result_df
    
    def generate_gt(self):
        # Here we only keep the rows with max one instance of each label
        def valid_labels(label_json_str):
            parsed_labels = self.process_labels(label_json_str)
            label_types = [label['type'] for label in parsed_labels]
            return label_types.count('A&P') <= 1 and label_types.count('HPI_Interval_Hx') <= 1

        valid_df = self.result_df[self.result_df.apply(
            lambda row: valid_labels(row['k_label']) and valid_labels(row['j_label']),
            axis=1
        )]
        # We also only keep the instances without conflict
        hpi_ap_mask = (valid_df['HPI_Interval_Hx'] == 1) & (valid_df['A&P'] == 1)
        valid_df = valid_df[hpi_ap_mask]
        valid_df = valid_df.merge(self.k_df[['note_text']], left_index=True, right_index=True)
        return valid_df
    
    def reorganize_outputs(self, outputs_df, answer_struct):
        # Create a dataframe that orders this data correctly
        new_rows = []
        for _, row in outputs_df.iterrows():
            k_labels = self.process_labels(row['k_label'])
            if "gpt_hx" in row:
                gpt_preds = json.loads(row['gpt_hx'])
                pain_status = answer_struct(**gpt_preds)
                start_pred_string = pain_status.start_string
                end_pred_string = pain_status.end_string
            else:
                start_pred_string = row['start_pred_string']
                end_pred_string = row['end_pred_string']
            for label in k_labels:
                # TODO: adapt to both labels
                if label['type'] == "HPI_Interval_Hx":
                    new_row = {
                        'note': row['note_text'],
                        'type': label['type'],
                        'start': label['start'],
                        'end': label['end'],
                        'start_pred_string': start_pred_string,
                        'end_pred_string': end_pred_string
                    }
                    new_rows.append(new_row)
        return pd.DataFrame(new_rows)