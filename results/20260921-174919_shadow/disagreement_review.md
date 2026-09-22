# Paired selector disagreement review

The shadow run executed GPT's choice only. JEV's alternative action was recorded but not executed, so this review does not claim its observed outcome.

Fourteen disagreements were between browser back and a visible link to the same category listing that browser back reached. The remaining disagreement was at hard repeat 3, step 14: GPT returned to Philosophy and JEV chose Home. Both were plausible routes toward Poetry, but only the GPT route was observed. None of these 15 cases demonstrates that JEV selected a more accurate next action.

| Run and step | GPT action | JEV action | Review |
|---|---|---|---|
| medium r1 step 3 | `back` | `click link "Travel"` | Same intended category listing; alternate action unexecuted |
| hard r1 step 3 | `click link "Travel"` | `back` | Same intended category listing; alternate action unexecuted |
| hard r1 step 5 | `click link "Travel"` | `back` | Same intended category listing; alternate action unexecuted |
| hard r1 step 7 | `click link "Travel"` | `back` | Same intended category listing; alternate action unexecuted |
| hard r1 step 12 | `back` | `click link "Philosophy"` | Same intended category listing; alternate action unexecuted |
| medium r2 step 3 | `back` | `click link "Travel"` | Same intended category listing; alternate action unexecuted |
| medium r2 step 5 | `back` | `click link "Travel"` | Same intended category listing; alternate action unexecuted |
| hard r2 step 5 | `click link "Travel"` | `back` | Same intended category listing; alternate action unexecuted |
| hard r2 step 10 | `click link "Philosophy"` | `back` | Same intended category listing; alternate action unexecuted |
| hard r3 step 3 | `back` | `click link "Travel"` | Same intended category listing; alternate action unexecuted |
| hard r3 step 10 | `click link "Philosophy"` | `back` | Same intended category listing; alternate action unexecuted |
| hard r3 step 14 | `back` | `click link "Home"` | Different intended route; counterfactual outcome unobserved |
| hard r3 step 17 | `back` | `click link "Poetry"` | Same intended category listing; alternate action unexecuted |
| medium r3 step 3 | `back` | `click link "Travel"` | Same intended category listing; alternate action unexecuted |
| medium r3 step 5 | `back` | `click link "Travel"` | Same intended category listing; alternate action unexecuted |

Source: `runs/<run_id>/steps.jsonl` for the nine run IDs in `runs.json`. The raw action, selected ID, confidence, and GPT-driven resulting URL are preserved there.
