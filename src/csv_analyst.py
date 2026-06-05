import plotly.express as px


def summarize_dataset(df):
    """
    Create a general summary of the uploaded CSV.
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": df.isnull().sum().to_dict()
    }

    return summary


def find_matching_column(question_lower, columns):
    """
    Find whether a column name appears in the user question.
    """

    for col in columns:
        if col.lower() in question_lower:
            return col

    cleaned_question = question_lower.replace("_", " ").replace("-", " ")

    for col in columns:
        cleaned_col = col.lower().replace("_", " ").replace("-", " ")

        if cleaned_col in cleaned_question:
            return col

    return None


def answer_csv_question(question, df):
    """
    Answer simple CSV questions using Pandas.

    Args:
        question: User question.
        df: Uploaded CSV as a Pandas DataFrame.

    Returns:
        answer: Text answer.
        fig: Plotly chart or None.
    """

    question_lower = question.lower()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Dataset summary
    if "summary" in question_lower or "summarize" in question_lower or "overview" in question_lower:
        summary = summarize_dataset(df)

        answer = f"""
Dataset Summary:
- Rows: {summary["rows"]}
- Columns: {summary["columns"]}
- Numeric columns: {", ".join(summary["numeric_columns"]) if summary["numeric_columns"] else "None"}
- Categorical columns: {", ".join(summary["categorical_columns"]) if summary["categorical_columns"] else "None"}
"""
        return answer, None

    # Missing values
    if "missing" in question_lower or "null" in question_lower:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if missing.empty:
            return "No missing values were found in the dataset.", None

        answer = "Missing Values:\n"

        for col, value in missing.items():
            answer += f"- {col}: {value}\n"

        return answer, None

    # Column list
    if "columns" in question_lower:
        answer = "Columns in the dataset:\n"

        for col in df.columns:
            answer += f"- {col}\n"

        return answer, None

    # Numeric columns
    if "numeric" in question_lower:
        if not numeric_cols:
            return "No numeric columns were found in the dataset.", None

        answer = "Numeric columns:\n"

        for col in numeric_cols:
            answer += f"- {col}\n"

        return answer, None

    # Categorical columns
    if "categorical" in question_lower or "category" in question_lower:
        if not categorical_cols:
            return "No categorical columns were found in the dataset.", None

        answer = "Categorical columns:\n"

        for col in categorical_cols:
            answer += f"- {col}\n"

        return answer, None

    # Total / sum
    if "total" in question_lower or "sum" in question_lower:
        matched_col = find_matching_column(question_lower, numeric_cols)

        if matched_col:
            total_value = df[matched_col].sum()
            answer = f"The total `{matched_col}` is {total_value:,.2f}."
            return answer, None

        if numeric_cols:
            answer = "Totals for numeric columns:\n"

            for col in numeric_cols:
                answer += f"- {col}: {df[col].sum():,.2f}\n"

            return answer, None

    # Average / mean
    if "average" in question_lower or "mean" in question_lower:
        matched_col = find_matching_column(question_lower, numeric_cols)

        if matched_col:
            avg_value = df[matched_col].mean()
            answer = f"The average `{matched_col}` is {avg_value:,.2f}."
            return answer, None

        if numeric_cols:
            answer = "Averages for numeric columns:\n"

            for col in numeric_cols:
                answer += f"- {col}: {df[col].mean():,.2f}\n"

            return answer, None

    # Highest / max
    if "highest" in question_lower or "maximum" in question_lower or "max" in question_lower:
        matched_col = find_matching_column(question_lower, numeric_cols)

        if matched_col:
            max_value = df[matched_col].max()
            answer = f"The highest value in `{matched_col}` is {max_value:,.2f}."
            return answer, None

        if numeric_cols:
            answer = "Maximum values for numeric columns:\n"
            for col in numeric_cols:
                answer += f"- {col}: {df[col].max():,.2f}\n"
            return answer, None

    # Lowest / min
    if "lowest" in question_lower or "minimum" in question_lower or "min" in question_lower:
        matched_col = find_matching_column(question_lower, numeric_cols)

        if matched_col:
            min_value = df[matched_col].min()
            answer = f"The lowest value in `{matched_col}` is {min_value:,.2f}."
            return answer, None

        if numeric_cols:
            answer = "Minimum values for numeric columns:\n"
            for col in numeric_cols:
                answer += f"- {col}: {df[col].min():,.2f}\n"
            return answer, None

    # Grouped chart: "show sales by region"
    if "by" in question_lower and categorical_cols and numeric_cols:
        category_col = find_matching_column(question_lower, categorical_cols)
        value_col = find_matching_column(question_lower, numeric_cols)

        if category_col and value_col:
            grouped = (
                df.groupby(category_col)[value_col]
                .sum()
                .reset_index()
                .sort_values(value_col, ascending=False)
                .head(10)
            )

            answer = f"Here is `{value_col}` grouped by `{category_col}`."

            fig = px.bar(
                grouped,
                x=category_col,
                y=value_col,
                title=f"{value_col} by {category_col}"
            )

            return answer, fig

    fallback_answer = """
I can answer basic CSV questions right now, such as:
- Give me a summary of the dataset
- What are the numeric columns?
- Which columns have missing values?
- What is the total of a numeric column?
- What is the average of a numeric column?
- Show a numeric column by a category column
"""

    return fallback_answer, None