The dataset used in this study was derived from the public Talk2Car_Command corpus, with each driving instruction manually labeled as Safe or Unsafe to form a custom subset and different lables for environmental conditions.
The model was trained on approximately 1000-1200 labeled commands and evaluated on a randomly selected test subset of 50 prompts, balanced across classes.

These results (≈ 95–98 % F1/Accuracy) reflect the controlled, low-noise nature of this test set rather than large-scale generalization.
This configuration was intentionally designed to validate the feasibility of context-aware prompt engineering in reproducible conditions.

Future work will include expanding the test corpus and releasing an open benchmark for broader robustness testing.
All reported metrics are internally consistent and reproducible via the provided scripts.
