# Known Limitations

## 1. Time-window parser uses only the first date reference

The time-window extractor reads the first quarter, half-year, or fiscal-year token it finds in a claim. Claims with two date anchors — such as "Revenue grew 20% in 2025 from the start of FY2024" — contain two year references. The parser detects the multi-reference case and abstains from filtering when "from" is present alongside multiple dates, defaulting to the full dataset range. However, single-anchor claims are still matched to whichever time token appears first, which may be the wrong one if the sentence structure is unusual (e.g., "By FY2025, revenue had grown since Q1 2023").

## 2. Derived and composite metrics are not verifiable

The claim verifier works by finding a column in the CSV whose name matches a term in the claim, then computing the percentage change in that column directly. Composite metrics — such as EBITDA (operating profit minus tax adjustments), revenue-per-employee (revenue ÷ headcount), or net new ARR (new ARR minus churned ARR) — cannot be verified even when all the constituent columns are present. These claims are correctly returned as Unverifiable, but with a generic "no column match" reason rather than a specific "this is a derived metric" explanation.

## 3. Claim extraction misses some pattern types

The extractor captures sentences that contain a percentage value with a direction word, a multiplier word (doubled/tripled/halved), a dollar magnitude with a direction word, a from-X-to-Y range, or a bare direction word followed by a number. It does not capture purely qualitative claims ("best-in-class NPS", "market leader in EMEA"), ordinal claims ("moved from third to first"), or claims expressed only as absolute values without a direction word ("EBITDA of $10M"). Such claims will not appear in the auto-extraction results.

## 4. Fuzzy column matching has an inherent accuracy/coverage trade-off

The current abstention threshold is 0.80 (rapidfuzz WRatio). This stops ambiguous matches such as "operating revenue" matching `revenue_usd_m` at 70% confidence from producing incorrect verdicts. However, raising the threshold also causes legitimate matches to abstain if the column name is an acronym, uses non-standard abbreviations, or differs significantly from the claim phrasing. Terms like "net new ARR", "blended ASP", or "logo churn" may fall below the threshold even when a matching column exists. Lowering the threshold to 0.65 would recover these matches at the cost of more false positives.
