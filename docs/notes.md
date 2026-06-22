flow:
1. Query gets analyzed.
2. Fixed intent table says: "debug usually gets 6000" or "explain usually gets 4000".
   That is fixed_budget.recommended.
3. ML features go into the Random Forest.
4. Random Forest votes for a budget bucket.
   That becomes raw_bucket.
5. We start with budget = raw_bucket.
6. If the model is low-confidence AND predicted below the fixed intent budget,
   bump it up one bucket.
7. Clamp the final budget inside the intent's min/max range.