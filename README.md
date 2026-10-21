# TruthLens

TruthLens is a small English text-classification prototype for an undergraduate
thesis demonstration. It uses the supplied ISOT Fake News Dataset, a word
TF-IDF representation, and Logistic Regression. It shows a REAL, FAKE, or
UNCERTAIN estimate, a model score, influential text features, held-out metrics,
RSS headlines, and a local history file.

This is a classifier, not a fact checker. It does not look up evidence or prove
whether a current story is true. The saved ISOT labels come from different
publisher groups: the dataset maintainers describe the real articles as
Reuters articles and the fake articles as collected from outlets considered
unreliable. As a result, a model can learn publisher, topic, or style patterns.
Its held-out score is not a reliable estimate for arbitrary new news.

## Thesis scope

The supplied proposal describes a multimodal transformer with text, images,
social context, early propagation, and advanced XAI. This implementation is a
text-only Logistic Regression baseline. It does not complete those proposed
components. A title that matches this prototype is:

**Explainable Text-Based Fake News Classification Using TF-IDF and Logistic
Regression**

If your department requires the original proposal title, discuss the narrowed
scope with your supervisor and describe this application as a baseline or
feasibility prototype. Do not describe it as a multimodal transformer or claim
that it detects early social-media propagation.

## Project files

~~~text
TruthLens/
├── app.py                       Streamlit defense demo
├── train_model.py               Dataset cleanup, training, evaluation, saving
├── config.py                    Shared project settings
├── requirements.txt             Tested Streamlit environment
├── run_truthlens.bat            Windows setup and launch
├── run_domain_version.bat       Windows preview of the HTML app
├── data/
│   ├── True.csv                 ISOT real-labelled articles
│   └── Fake.csv                 ISOT fake-labelled articles
├── history.csv                  Local-only Streamlit analysis history
└── domain_version/
    ├── app.py                   Flask web server for the HTML version
    ├── predictor.py             Model loading and feature explanations
    ├── news.py                  Cached RSS reader
    ├── config.py                Standalone deployment settings
    ├── requirements.txt         Domain-host dependencies
    ├── Procfile                 Production WSGI launch command
    ├── models/
    │   ├── model.joblib         Trained TF-IDF + Logistic Regression pipeline
    │   └── model_metrics.json   Generated evaluation report
    ├── static/
    │   ├── app.js               Browser interactions
    │   └── style.css            Lightweight responsive design
    └── templates/
        └── index.html           HTML page
~~~

## Install and run the defense version

Use Python 3.13. The included Windows launcher prepares a virtual environment,
installs the pinned packages, trains once if the model is missing, and opens
the app:

~~~powershell
.\run_truthlens.bat
~~~

For a manual setup:

~~~powershell
$python = "$env:LOCALAPPDATA\Programs\TruthLens\Python313\python.exe"
& $python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

If Python is installed in another location, change the first line to that
Python 3.13 executable. The Windows launcher searches common install locations
automatically.

Streamlit prints the local URL after startup. Keep its terminal open during the
demonstration. If the model files already exist, skip the training command.

## Train and evaluate

The training script reads data/True.csv and data/Fake.csv. It combines each
article's title and body, applies the same text cleanup used by the app, removes
empty rows, removes exact duplicate cleaned texts, and drops conflicting
duplicates if any exist. It then makes a reproducible, stratified 80/20
train/test split. TF-IDF is fitted on the training split only.

The model uses word unigrams and bigrams followed by Logistic Regression. The
test report includes accuracy, macro precision, macro recall, macro F1, a
confusion matrix, per-class values, and the fraction of test inputs that meet
the application's fixed 60% score cutoff. Metrics are written to
domain_version/models/model_metrics.json; the trained pipeline is written to
domain_version/models/model.joblib.

The 60% cutoff is a simple UI rule, not a calibrated probability threshold.
“Accuracy when not uncertain” is reported separately from accuracy across all
test examples. Confusion-matrix results use the best REAL/FAKE class for every
test item before the UNCERTAIN rule.

## What the explanation means

For the submitted text, the app converts recognized terms to TF-IDF values and
multiplies them by the trained Logistic Regression coefficients. It displays
the largest positive and negative feature contributions. These terms explain
how the model score was formed; they do not establish that the claim is true
or false. The score itself is not calibrated as a probability of truth.

## Latest-news demonstration

The app fetches a few headlines from the RSS URLs in domain_version/config.py
when requested. Results are cached for 30 minutes and can be manually
refreshed. The app can score a headline, but RSS items are not training data,
evaluation data, or fact-checks.

## Separate HTML version for a domain

The self-contained domain_version/ folder is the second interface: a plain
HTML/CSS/JavaScript page served by a small Flask application. It includes its
own inference code, generated model, metrics, RSS reader, and hosting
requirements. It does not save visitor submissions or provide the local
history page.

To preview it on Windows, run:

~~~powershell
.\run_domain_version.bat
~~~

Then open http://127.0.0.1:5000.

For hosting, deploy the contents of domain_version/ as a Python WSGI
application, install requirements.txt, and use the start command:

~~~text
waitress-serve --listen=0.0.0.0:$PORT app:app
~~~

The included Procfile contains that command. Waitress supports Windows and
Linux. The models/ folder must include
model.joblib and model_metrics.json before deployment. The CSV training data
is not needed by the running web app and should not be uploaded with the domain
version. The hosting provider must support Python/WSGI and custom-domain
mapping; DNS and HTTPS setup are performed in that provider's dashboard and
depend on the selected host.

Do not use Flask's built-in local development server as the public production
server. Flask recommends a dedicated WSGI server or hosting platform for
deployment; the included deployment command uses Waitress.

## Defense demonstration

1. Start the Streamlit app and show the project scope on Home.
2. Open Model Performance and explain the split, all four metrics, and the
   confusion matrix.
3. Analyze a complete English news article. Explain the score and top terms.
4. Show that an unsupported or very short input can produce UNCERTAIN.
5. Open Latest News, refresh RSS, then score one headline.
6. Open History and explain that this version saves the text locally in
   history.csv.
7. State the ISOT source-label and publisher-distribution limitations.
8. State which parts of the proposal are outside this prototype.

## References

- ISOT Research Lab, University of Victoria, [ISOT Fake News Dataset and
  dataset notes](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf).
- Streamlit, [basic concepts](https://docs.streamlit.io/get-started/fundamentals/main-concepts)
  and [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
- Flask, [deployment to production](https://flask.palletsprojects.com/en/stable/deploying/).
