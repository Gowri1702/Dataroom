import pytest
from src.router import route_question


class TestRouteQuestion:
    def test_pdf_document_keyword(self):
        assert route_question("What does the document say?") == "pdf"

    def test_pdf_risks(self):
        assert route_question("What are the key risks in the report?") == "pdf"

    def test_pdf_summary(self):
        assert route_question("Summarize the main takeaways.") == "pdf"

    def test_pdf_financial_highlights(self):
        assert route_question("What are the financial highlights?") == "pdf"

    def test_pdf_page_reference(self):
        assert route_question("What does page 5 say about strategy?") == "pdf"

    def test_pdf_narrative(self):
        assert route_question("Describe the narrative in the report.") == "pdf"

    def test_csv_total(self):
        assert route_question("What is the total revenue?") == "csv"

    def test_csv_average(self):
        assert route_question("What is the average ARR?") == "csv"

    def test_csv_columns(self):
        assert route_question("What are the columns in the dataset?") == "csv"

    def test_csv_missing_values(self):
        assert route_question("Which columns have missing values?") == "csv"

    def test_csv_max(self):
        assert route_question("What is the maximum MRR?") == "csv"

    def test_csv_min(self):
        assert route_question("What is the lowest ARR?") == "csv"

    def test_csv_grouped(self):
        assert route_question("Show revenue by region.") == "csv"

    def test_csv_summary(self):
        assert route_question("Give me a summary of the dataset.") == "csv"

    def test_both_verify(self):
        assert route_question("Verify the revenue claim against the data.") == "both"

    def test_both_compare(self):
        assert route_question("Compare the report with the CSV.") == "both"

    def test_both_match(self):
        assert route_question("Does the PDF match the spreadsheet?") == "both"

    def test_both_claim(self):
        assert route_question("Are the claims in the document consistent?") == "both"

    def test_both_validate(self):
        assert route_question("Validate the growth claim against the data.") == "both"

    def test_both_beats_csv(self):
        assert route_question("Verify the total revenue claim.") == "both"

    def test_default_is_pdf(self):
        assert route_question("This question has no matching keyword.") == "pdf"

    def test_case_insensitive(self):
        assert route_question("WHAT IS THE TOTAL REVENUE?") == "csv"
