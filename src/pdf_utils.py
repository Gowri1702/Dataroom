import fitz


def extract_pdf_text(uploaded_pdf):

    if uploaded_pdf is None:
        return "", []

    pdf_bytes = uploaded_pdf.read()

    pages = []
    full_text = ""

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_doc:
        for page_num, page in enumerate(pdf_doc):
            text = page.get_text()

            page_data = {
                "page_number": page_num + 1,
                "text": text
            }

            pages.append(page_data)

            full_text += f"\n\n--- Page {page_num + 1} ---\n{text}"

    return full_text, pages