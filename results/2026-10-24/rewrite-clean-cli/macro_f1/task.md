Task type: classification
Primary metric: macro-F1
Target column: outcome_strange
ID column: opaque_key
Train: `inputs/unfamiliar/learning_table.csv`
Test: `inputs/unfamiliar/unknown_rows.csv`
Sample: `inputs/unfamiliar/expected_format.csv`
Dictionary: `inputs/unfamiliar/column_notes.csv`