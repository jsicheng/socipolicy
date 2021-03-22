# Socipolicy
Predicting Covid-19 Policy Compliance using Twitter Data

Currently supported dataset:
- Mask Wearing Likelihood
- Covid-19 Vaccine Acceptance

## Setup
```
cd src
pip install -r requirements.txt
```
## Run on local host
To run on localhost:
```
python manage.py runserver
```
The website will be deployed at http://localhost:8000

The website is setup to deploy on Heroku, however since Twitter blocks all AWS IPs, snscrape will [not work](https://github.com/JustAnotherArchivist/snscrape/issues/79) on Heroku.
