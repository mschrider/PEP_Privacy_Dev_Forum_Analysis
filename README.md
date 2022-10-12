# PEP_Privacy_Dev_Forum_Analysis
Repository for code related to the COS 535 Project - Analysis of Reddit Privacy Questions and Answers Adults
## Project Team Members
Jonathan Parsons  
Oyebanjo (Banjo) Ogunlela  
Michael Schrider

## Project Description
Reddit covers various software development topics. In this project, you are expected to first
conduct a literature review on the current approaches and results related to the analysis of
developers’ forums and discuss their results, and their shortcomings. Then propose an
improvement to the current approaches to create insights about the concerns and privacy
challenges of developers on Reddit. You are expected to extract information about developers'
questions related to regulations; privacy engineering practices; the solutions proposed by other
developers; the most answered questions; and the most accepted/rejected answers, etc. You will
also examine what the trends of questions or answers are pre- vs post-GDPR, CCPA and other
similar regulations and then report the results. You are expected to at least evaluate 100 Q/As in
the last five years. In addition to the paper, you are expected to make your dataset, tool, all other
relevant artifacts available on GitHub.

## Git Project Scope
This git project is intended to keep a record of and allow easy reproduction of the data and analysis used in the Reddit privacy analysis. The three subreddits targeted for analysis are /r/androiddev, /r/iOSprogramming, and /r/webdev. Static datasets for these three subreddits used for the Reddit privacy analysis are included and can be used to directly reproduce this project's work.

Code within this project can potentially be leveraged to analyze privacy behavior of other subreddits but compatibility is not guaranteed.

# Dependencies and Prerequisites - TBD
## Python Library Dependencies - TBD
This project is built from the following python libraries:
* [Python Reddit API Wrapper (PRAW)](https://praw.readthedocs.io/en/stable/getting_started/authentication.html "Python Reddit API Wrapper (PRAW)")
* [Pandas](https://pandas.pydata.org/ "Pandas")
* [matplotlib](https://matplotlib.org/ "matplotlib")
## Reddit Account and Application Registration
For fetching up to date Reddit posts, this project requires a read only connection with Reddit. This connection is not required if only static datasets from github are used. 

The suggested way to setup a connection is using [PRAW OAuth](https://praw.readthedocs.io/en/stable/getting_started/authentication.html) from PRAW for a registered Reddit Script Applicaiton.

In order to register a script application with reddit, a Reddit account is required. Reddit applications can be registered at: https://www.reddit.com/prefs/apps/

To use PRAW OAuth, connection details based on the registered account/application need to be provided, an easy way to do this is in the praw.ini file ([How to setup configuration file](https://github.com/mschrider/PEP_Privacy_Dev_Forum_Analysis/edit/main/README.md#setup-configuration-file)).

# How to Use

## Directly Running with Static Datasets - TBD
The simpliest method to directly reproduce the outputs of this project using the static datasets in github is to run the following:   
```python
import reddit_privacy_analysis 
reddit_privacy_analysis.run_project(fetch_data=False)
```
This will generate several matplotlib objects and produce csv outputs in the current python working directory.

## Setup Configuration File
Setting up a After cloning or otherwise getting this project down to use, the praw.ini file needs to be updated with login information.

The *client_id* and *client_secret* parameters are obtained when registering a Reddit application and if you have a registered app can be found at https://www.reddit.com/prefs/apps/.

The value in *user_agent* does not matter.

The *username* and *password* parameters need to be associated with a Reddit account that is a devloper of the app referenced in *client_id*.

This ini file will contain your Reddit password, so be sure to **NOT** share a filled in praw.ini unintentionally. 
```
[DEFAULT]
client_id=YOUR_REGISTERD_APP_ID
client_secret=YOUR_REGISTERD_APP_SECRET_ID
user_agent=USER_AGENT_NAME
username=REDDIT_APP_DEVELOPER_ACCOUNT_USER_NAME
password=REDDIT_APP_DEVELOPER_ACCOUNT_PASSWORD
```
Other methods for providing the connection information can be used; see [PRAW OAuth](https://praw.readthedocs.io/en/stable/getting_started/authentication.html) for details.


## Loading Data
### Load Existing Dataset - TBD
This project includes pre-pulled static datasets, these can be loaded into Pandas dataframes.
```python
data = reddit_privacy_analysis.load_dataset('androiddev')
```

### Query Reddit for New Data - TBD
Up to date Reddit can be fetched. Getting current data requires a connection with Reddit, see [How to setup configuration file](https://github.com/mschrider/PEP_Privacy_Dev_Forum_Analysis/edit/main/README.md#setup-configuration-file) for instructions on how to enable an authorized connection with Reddit. Newly fetched data will need to be cleansed see [Cleansing Fetched Data](https://github.com/mschrider/PEP_Privacy_Dev_Forum_Analysis/edit/main/README.md#cleansing-fetched-data---tbd)
```python
data = reddit_privacy_analysis.fetch_dataset('androiddev')
```

### Cleansing Fetched Data - TBD

## Producing Analysis Outputs - TBD
Details on the implementation of the analysis are located in [Data Analysis](https://github.com/mschrider/PEP_Privacy_Dev_Forum_Analysis/edit/main/README.md#data-analysis---tbd).


# Data Analysis - TBD

## ERD and Format of Reddit Data - TBD

## Data Cleansing - TBD

## Sentiment Analysis - TBD
### Logic Flow Diagrams - TBD

## Other Analysis - TBD
### Logic Flow Diagrams - TBD
