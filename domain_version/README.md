# TruthLens HTML domain version

This folder is a self-contained web deployment of the TruthLens student
prototype. It uses HTML/CSS/JavaScript with a small Flask JSON service and the
trained model in models/. It does not need the CSV training data to run and
does not retain visitor-submitted text.

## Local preview on Windows

From the TruthLens project folder:

~~~powershell
.\run_domain_version.bat
~~~

Or install the dependencies and run the local preview directly:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r domain_version/requirements.txt
.\.venv\Scripts\python.exe domain_version/app.py
~~~

Open http://127.0.0.1:5000. The local Flask server is only for preview.

## Hosting

Deploy this folder as the application's root on a Python WSGI hosting service.
Install the dependencies from requirements.txt; use app:app as the WSGI
application and start the service with:

~~~text
waitress-serve --listen=0.0.0.0:$PORT app:app
~~~

The Procfile contains this command. Before deployment, ensure
models/model.joblib and models/model_metrics.json exist. Running
python train_model.py from the parent TruthLens project folder writes both
files into this folder's models/ directory. The dataset CSVs are only needed
for training.

The hosting provider needs to support Python/WSGI apps and custom-domain
mapping. Attach the domain and set DNS records in that provider's dashboard;
the exact records depend on the host. This package is prepared for deployment,
but it has not been uploaded to a host or connected to a real domain.

This is an academic demonstration, without accounts, a database, rate limits,
or external evidence verification. Flask's built-in server is only for local
preview; use the included Waitress WSGI command for public hosting.
