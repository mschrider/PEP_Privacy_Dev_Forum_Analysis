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

# Dependencies and Prerequisites
## Python Library Dependencies
This project is built from the following python libraries:
* [Python Reddit API Wrapper (PRAW)](https://praw.readthedocs.io/en/stable/getting_started/authentication.html "Python Reddit API Wrapper (PRAW)")
* [Pandas](https://pandas.pydata.org/ "Pandas")

## Reddit Account and Application Registration
This project requires a read only connection with Reddit to pull posts from the various sub-reddits. The suggested way to setup a connection is using [PRAW OAuth](https://praw.readthedocs.io/en/stable/getting_started/authentication.html) from PRAW for a registered Reddit Script Applicaiton.

In order to register a script application with reddit, a Reddit account is required. Reddit applications can be registered at: https://www.reddit.com/prefs/apps/

To use PRAW OAuth, connection details based on the registered account/application need to be provided, an easy way to do this is in the praw.ini file ([How to setup configuration file](https://github.com/mschrider/PEP_Privacy_Dev_Forum_Analysis/edit/main/README.md#setup-configuration-file)).

# How to Use

## Setup Configuration File
After cloning or otherwise getting this project down to use, the praw.ini file needs to be updated with login information.

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
