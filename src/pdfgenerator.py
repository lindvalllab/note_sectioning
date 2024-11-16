import json
import numpy as np
import tempfile
from weasyprint import HTML

class PDFGenerator:
    """
    A class to generate an HTML page with notes, highlighted sections for each researcher, and overlap sections.
    It converts the HTML to PDF and deletes the temporary HTML file afterwards.
    """

    def __init__(self, k_df, j_df, result_df):
        """
        Initialize the HTMLToPDFGenerator with two dataframes (k_df, j_df) and file names.

        Args:
        - k_df: DataFrame containing researcher K's labels and note_text.
        - j_df: DataFrame containing researcher J's labels and note_text.
        - result_df: DataFrame containing overlap scores and other related data.
        """
        self.k_df = k_df.sort_index()
        self.j_df = j_df.sort_index()
        self.result_df = result_df
        self.html_content = ""

    def create_html_content(self):
        """
        Create the HTML content with notes, highlighted sections, and a legend.
        """
        assert self.k_df['note_text'].equals(self.j_df['note_text']), "The 'note_text' columns are not identical."

        # Begin the HTML content
        self.html_content += "<html><head><style>"
        self.html_content += "body { font-family: Arial, sans-serif; line-height: 1.5; margin: 20px; }"
        self.html_content += "p { text-align: justify; }"  # Justify text
        self.html_content += ".highlight-k { background-color: yellow; }"  # Researcher K's highlights
        self.html_content += ".highlight-j { background-color: lightgreen; }"  # Researcher J's highlights
        self.html_content += ".highlight-overlap { background-color: lightblue; }"  # Overlap highlights
        # Style for label type
        self.html_content += (
            ".label-type {"
            "display: inline-block; font-size: 9px; font-weight: bold;"
            "background-color: #E0E0E0; border: 1px solid black; padding: 2px 4px;"
            "border-radius: 3px; text-align: center; margin-right: 5px;"
            "vertical-align: middle; margin-left: 5px}"
        )
        self.html_content += ".legend { font-size: 12px; margin-bottom: 10px; }"  # Style for legend
        self.html_content += ".overlap-scores { font-size: 12px; font-style: italic; margin-left: 10px; }"  # Style for overlap scores
        self.html_content += ".page-break { page-break-before: always; }"  # CSS for page break
        self.html_content += "</style></head><body>"

        for index, row in self.k_df.iterrows():
            # Add a page break before each new note (except the first one)
            if index > 0:
                self.html_content += "<div class='page-break'></div>"
            
            self.html_content += self.add_legend()

            self.html_content += f"<h2>Note {index}"
            # If only one person annotated, no overlap scores are available
            if index in self.result_df.index:
                overlap_scores = self.result_df.loc[index]
                self.html_content += self.display_overlap_scores(overlap_scores)
            else:
                self.html_content += "<span class='overlap-scores'>HPI Interval Hx: N/A, A&P: N/A</span>"
            self.html_content += "</h2>"

            note_text = row['note_text']
            self.html_content += self.display_note_with_highlights(index, note_text)

        self.html_content += "</body></html>"

    def add_legend(self):
        """
        Create a color legend to explain the highlight colors on each page.
        """
        legend = """
        <div class="legend">
            <span class="highlight-k">K's annotations</span> - Yellow &nbsp;&nbsp;
            <span class="highlight-j">J's annotations</span> - Green &nbsp;&nbsp;
            <span class="highlight-overlap">Overlap</span> - Blue
        </div>
        """
        return legend

    def display_overlap_scores(self, overlap_scores):
        """
        Display the overlap scores next to the note title in smaller font.
        Args:
        - overlap_scores: A Series with overlap scores for the current note.
        Returns:
        - A string with the overlap scores formatted in HTML.
        """
        scores_html = "<span class='overlap-scores'>("
        HPI, AP = overlap_scores['HPI_Interval_Hx'], overlap_scores['A&P']
        HPI = round(HPI, 3) if not np.isnan(HPI) else "N/A"
        AP = round(AP, 3) if not np.isnan(AP) else "N/A"
        scores_html += f"HPI Interval Hx: {HPI}, A&P: {AP}"
        scores_html += ")</span>"
        return scores_html

    def display_note_with_highlights(self, index, note_text):
        """
        Generate the HTML for a single note, with highlights from both researchers.

        Args:
        - index: The index of the note.
        - note_text: The complete note text.

        Returns:
        - A string with the HTML for the note, including highlighted sections.
        """
        k_parsed_annotations = self.process_labels(self.k_df.loc[index, 'label'])
        j_parsed_annotations = self.process_labels(self.j_df.loc[index, 'label'])
        highlighted_text = self.highlight_text(note_text, k_parsed_annotations, j_parsed_annotations)
        return f"<p>{highlighted_text}</p>"

    def highlight_text(self, note_text, k_annotations, j_annotations):
        """
        Highlight the sections of the note text based on the annotations,
        and include the label type at the beginning of each highlighted section.
        
        Args:
        - note_text: The complete note text.
        - k_annotations: Annotations from researcher K.
        - j_annotations: Annotations from researcher J.
        
        Returns:
        - A string with the note text, with highlighted sections wrapped in HTML <span> tags.
        """
        # Collect all unique start and end positions from both annotations
        boundaries = set()
        for ann in k_annotations:
            boundaries.add(ann['start'])
            boundaries.add(ann['end'])
        for ann in j_annotations:
            boundaries.add(ann['start'])
            boundaries.add(ann['end'])

        # Add boundaries for the start and end of the text
        boundaries.add(0)
        boundaries.add(len(note_text))

        sorted_boundaries = sorted(boundaries)
        
        # Create intervals between consecutive boundaries
        intervals = []
        for i in range(len(sorted_boundaries) - 1):
            intervals.append((sorted_boundaries[i], sorted_boundaries[i+1]))
        
        highlighted_text = ""
        
        for start, end in intervals:
            segment = note_text[start:end]
            # Determine if the segment is annotated by K, J, or both
            in_k = [ann['type'] for ann in k_annotations if ann['start'] <= start and ann['end'] >= end]
            in_j = [ann['type'] for ann in j_annotations if ann['start'] <= start and ann['end'] >= end]
            
            # Assign label and CSS class based on annotations
            if in_k and in_j:
                # Find common labels
                common_labels = set(in_k).intersection(set(in_j))
                if common_labels:
                    label_type = common_labels.pop()  # Use the common label
                else:
                    # If no common label, concatenate labels or handle as needed
                    label_type = "/".join(sorted(set(in_k + in_j)))
                css_class = 'highlight-overlap'
            elif in_k:
                label_type = in_k[0]  # Use the first label from K's annotations
                css_class = 'highlight-k'
            elif in_j:
                label_type = in_j[0]  # Use the first label from J's annotations
                css_class = 'highlight-j'
            else:
                label_type = None
                css_class = None
            
            if label_type:
                label_tag = f"<span class='label-type'>{label_type}: </span>"
                highlighted_text += f"{label_tag}<span class='{css_class}'>{segment}</span>"
            else:
                highlighted_text += segment
        
        return highlighted_text

    def process_labels(self, json_string):
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
            labels = item["labels"]
            start = item["start"]
            end = item["end"]
            for label in labels:
                sections.append({"start": start, "end": end, "type": label})
        return sections

    def convert_to_pdf(self, output):
        """
        Convert the generated HTML content to a PDF using WeasyPrint and delete the HTML file.
        """
        # Create a temporary HTML file, and use it to generate the PDF
        with tempfile.NamedTemporaryFile(suffix=".html") as temp_html:
            self.create_html_content()
            temp_html.write(self.html_content.encode('utf-8'))
            temp_html_path = temp_html.name
            HTML(temp_html_path).write_pdf(output)