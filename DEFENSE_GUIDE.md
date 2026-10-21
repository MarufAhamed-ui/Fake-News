# TruthLens defense guide

## One-sentence description

TruthLens is a lightweight English text-classification baseline that learns
TF-IDF word patterns from the ISOT dataset, predicts a REAL or FAKE class, and
shows which recognized text features contributed most to that score.

## Explain the workflow

1. Combine the ISOT title and article body.
2. Lowercase the text, remove URLs and punctuation, and collapse whitespace.
3. Remove exact duplicate cleaned texts before splitting.
4. Make an 80/20 stratified train/test split with a fixed random seed.
5. Fit TF-IDF and Logistic Regression on the training split.
6. Report accuracy, macro precision, macro recall, macro F1, and the confusion matrix on the held-out test split.
7. For a new input, compute the highest class score. Display UNCERTAIN below the fixed 60% cutoff or when no vocabulary terms are recognized.
8. Multiply input TF-IDF values by model coefficients to show local feature contributions.

## Suggested answers to common questions

**Why TF-IDF and Logistic Regression?**  
They are fast on a normal laptop, form a useful text baseline, expose readable
feature weights, and are easier to explain than a large neural model.

**Does REAL mean verified true?**  
No. It means the classifier's REAL class scored higher. The model does not
retrieve evidence or perform fact verification.

**What does the score mean?**  
It is a class score from Logistic Regression. This prototype does not calibrate
the score as the probability that an article is true or false.

**How are the highlighted words selected?**  
Each recognized input feature's TF-IDF value is multiplied by its learned
Logistic Regression coefficient. Positive contributions move the FAKE class
score upward; negative contributions move it toward REAL. Contributions are
model explanations, not proof.

**Why does the evaluation use a random split?**  
It gives a simple reproducible held-out test on deduplicated records from the
same dataset distribution. It does not test generalization to new publishers,
topics, languages, or current events.

**What is the main evaluation limitation?**  
ISOT's two files come from different publisher groups. Labels are inherited
from file/source membership, not independently checked for every claim. The
classifier may learn source or writing-style cues. An external dataset with
documented article-level fact-check labels is needed for a stronger
generalization claim.

**Is this the multimodal transformer described in the proposal?**  
No. This prototype contains no image model, social graph, propagation-time
features, Transformer, SHAP, or LIME. It is a text-only baseline. The thesis
scope/title must be approved or the missing components must be implemented
before claiming the original proposal is complete.

**What does real-time mean here?**  
The local model can score text quickly, and the application can retrieve
current RSS headlines on demand. It does not monitor social platforms or
identify misinformation during early propagation.

**What would be the next research step?**  
Evaluate against a source-independent and time-separated external test set.
Then add one proposed modality at a time and compare it with this baseline,
while keeping the data split and explanations auditable.

## Claims to avoid

- The model proves a story is fake.
- The system is 100% accurate.
- The result is an independently verified truth probability.
- The prototype is multimodal, transformer-based, or detects early spread.
- The held-out ISOT score predicts performance on all current news.
