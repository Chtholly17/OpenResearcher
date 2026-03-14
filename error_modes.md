# GAIA Trajectory Error Modes

Reference document defining error modes observed in GAIA benchmark inference trajectories.

## Error Mode Definitions

| ID | Error Mode | Description |
|----|-----------|-------------|
| E1 | Insufficient Search Depth | Too few searches (typically 1) before answering; no query refinement or follow-up searches |
| E2 | Hallucination from Incomplete Context | Specific facts/numbers in final answer not found in any tool result |
| E3 | Incorrect Search Query | Queries too vague, wrong terminology, or missing key terms that would find the answer |
| E4 | Misinterpretation of Search Results | Correct info was present in search results but model picked wrong value/entity |
| E5 | Reasoning / Logic Error | Arithmetic, logic, or multi-step reasoning mistakes despite having correct facts |
| E6 | Unsupported Numerical Estimation | Specific numbers produced without any supporting search data |
| E7 | Failure to Navigate to Specific Documents | Cannot access required PDFs, GitHub issues, specific web pages, or databases |
| E8 | Runtime Error / Timeout | Trajectory ended with `status != "success"` (error, timeout, max turns exceeded) |
| E9 | Other / Unclear | Doesn't fit above categories clearly |

## Detailed Descriptions and Examples

### E1: Insufficient Search Depth

The model performs only 1 search (or very few) and then immediately answers, without refining queries or exploring further. Often the single search doesn't return the specific information needed.

**Example 1**: qid=0, "Where was the Finding Nemo fish found as nonnative species (USGS zip codes)?"
- Model did 1 search: `nonnative species of clownfish USGS zip codes`
- Got generic results, then hallucinated zip codes `33677, 92662, 96810` instead of searching more specifically on USGS NAS database
- Correct answer: `34689`

**Example 2**: qid=4, "How many studio albums by Mercedes Sosa between 2000-2009?"
- Model did 1 search, got a partial discography, answered `5` instead of drilling into specific album details
- Correct answer: `3`

### E2: Hallucination from Incomplete Context

The model's final answer contains specific facts, numbers, or claims that do not appear in any of the tool results it received. The model fabricates details to fill gaps.

**Example**: qid=0 (same as above) - The zip codes `33677, 92662, 96810` were not in any search result. The model invented plausible-sounding but incorrect zip codes.

### E3: Incorrect Search Query

The search queries use wrong terms, are too broad, or miss the key specifics needed to find the answer.

**Example**: Searching for `"Mercedes Sosa albums"` instead of `"Mercedes Sosa discography studio albums 2000 2009"` - the vague query returns a general page that doesn't clearly distinguish studio albums from compilations.

### E4: Misinterpretation of Search Results

The correct information was present in the search results, but the model extracted or interpreted the wrong value.

**Example**: A search result lists multiple values/entities, and the model picks the wrong one (e.g., confusing a compilation album with a studio album, or picking the wrong person from a list of names).

### E5: Reasoning / Logic Error

The model has the right facts from searches but makes arithmetic mistakes, logical errors, or incorrect multi-step reasoning.

**Example**: Correctly finding a list of albums but miscounting them, or correctly finding two numbers but computing the wrong arithmetic operation on them.

### E6: Unsupported Numerical Estimation

The model produces a specific number in its answer without any search data backing it up. Different from E2 in that the model appears to be estimating/guessing rather than misremembering.

**Example**: When asked for a specific count or measurement, the model provides a number based on "general knowledge" without ever searching for the actual data.

### E7: Failure to Navigate to Specific Documents

The question requires accessing a specific resource (PDF, GitHub issue, database page, specific URL) and the model either doesn't attempt to navigate there or fails to extract the needed information from it.

**Example**: Question requires reading a specific PDF attachment or GitHub issue, but the model only does web searches and never uses `browser.open` to access the specific resource.

### E8: Runtime Error / Timeout

The trajectory has `status != "success"` - the agent hit an error, timeout, or max turn limit before producing a final answer.

### E9: Other / Unclear

The error doesn't clearly fit any of the above categories, or the trajectory is too convoluted to classify.

## Classification Guidelines

- Each trajectory can have **multiple** error modes (e.g., E1 + E2 often co-occur).
- One error mode should be marked as the **primary** cause of failure.
- E8 should be classified from trajectory metadata (`status` field), not from LLM analysis.
- When E1 and E2 co-occur, the primary is usually E1 (insufficient search led to hallucination).
- E6 is a specific subtype of E2 focused on numerical values with no search support at all.
