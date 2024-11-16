import json

class ErrorLogger:
    """
    A class for logging errors based on threshold criteria for HPI_Interval_Hx and A&P.
    """

    def __init__(self, df_clean):
        """
        Initialize the ErrorLogger with the cleaned DataFrame.
        
        Args:
        - df_clean: The DataFrame containing cleaned data.
        """
        self.df_clean = df_clean  # DataFrame to work with

    @staticmethod
    def log_lines_to_write(row, index, lines_to_write, label_type):
        """
        Log lines for rows where HPI_Interval_Hx or A&P is below a threshold.

        Args:
        - row: A row from the DataFrame.
        - index: The row index.
        - lines_to_write: A list to collect lines for writing to a file.
        - label_type: The type of label ('HPI_Interval_Hx' or 'A&P').
        """
        # Log the row index and value of the label type
        lines_to_write.append(f"\n{'_'*20} \n Row {index} - {label_type}: {round(row[label_type], 3)}\n")
        
        try:
            # Parse k_label and j_label from the row
            k_label = json.loads(row['k_label']) if isinstance(row['k_label'], str) else row['k_label']
            j_label = json.loads(row['j_label']) if isinstance(row['j_label'], str) else row['j_label']
        except json.JSONDecodeError:
            print(f"Error parsing labels at row {index}")
            return

        lines_to_write.append(f"\n KS indexes:")
        for dic in k_label:
            if dic['labels'] == [label_type]:
                lines_to_write.append(f"\n {dic['start']} - {dic['end']}")

        lines_to_write.append(f"\n\n JB indexes:")
        for dic in j_label:
            if dic['labels'] == [label_type]:
                lines_to_write.append(f"\n {dic['start']} - {dic['end']}\n")

    def log_errors(self, filename):
        """
        Write errors to a file based on threshold checks for HPI_Interval_Hx and A&P.
        
        Args:
        - filename: The name of the file where errors will be logged.
        """
        # Filter rows where either HPI_Interval_Hx or A&P is below 0.8
        errors_df = self.get_error_df()
        
        with open(filename, "w") as file:
            # Iterate through each row with errors and log relevant information
            for index, row in errors_df.iterrows():
                hpi_interval_hx_below_threshold = row['HPI_Interval_Hx'] < 0.8
                ap_below_threshold = row['A&P'] < 0.8

                lines_to_write = []

                # Log HPI_Interval_Hx-related errors
                if hpi_interval_hx_below_threshold:
                    self.log_lines_to_write(row, index, lines_to_write, 'HPI_Interval_Hx')

                # Log A&P-related errors
                if ap_below_threshold:
                    self.log_lines_to_write(row, index, lines_to_write, 'A&P')

                # Write the collected lines to the file
                if lines_to_write:
                    file.write("".join(lines_to_write))

    def get_error_df(self):
        return self.df_clean[(self.df_clean['HPI_Interval_Hx'] < 0.8) | (self.df_clean['A&P'] < 0.8)]