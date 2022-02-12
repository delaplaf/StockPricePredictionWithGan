# Stock Price Prediction With WGAN-GP on our own dataset



## Why this project? 

First of all, it was the work of two students (Florian Delaplace, Bilel Hakem) who were simply looking to find out more about GAN. Usually, we often see them used for images however following Boris B’s article [1] we saw other possible tracks. Unfortunately, little code was provided but the subject was of great interest to us: the prediction of the price of the stock exchange using gan (more precisely WGAN-GP). This article aims to tell you about our journey to try to reproduce these results. \
For starters, a little state of the art. Looking a little, we quickly came across [4] who took over the work of Boris B and provided a functional code. That made our job much easier. But the code still required no cleaning and we added a lot of functionality to it, whether for training models or evaluating results.

## Project Structure

- New/ : all useful codes for the scrapping of news and for the analysis of feelings on articles.
- Data/DataFacebook.csv : final data before preprocessing
- Data/dataPreprocessed/ : npy files containing data after preprocessing
- Code/all_models/ : All models used to compare results.
- Code/all_models/wgan_gp/ : Implementation of a WGAN-GP
- Code/preprocessing.py : all the preprocessing needed
- Code/test_pred.py : plot and metric for model evaluation
- endToEndGan.ypinb : notebook to easily use this project
- all models have an associated test script

## Collecting data

First of all, we chose to collect our own data. We took an interest in a big company known to everyone: Meta. In order to do this, financial indicators have been collected online and are freely available. To this, we added a sentiment analysis on the news that we collected on the Seeking Alpha platform. \
For scrapping we used a pretty basic code based on Selenium and Beautiful Soup. Basically we just went through all the news pages about Meta and retrieved the titles and dates of the various articles. \
For the analysis of feelings, we just retrieved the FinBert preloaded model and used it for the prediction as indicated in the excellent tutorial [7].


## The preprocessing

To be quick, we simply extracted financial technical indicators and also used different transformed Fourier as features. For a more detailed explanation, I invite you to read the work of Boris B cited above. \
Then, we fill in all the missing values at a date t by the average of the value at date t-1 and the value at date t+1. \
Finally we normalized the data with the sklearn MinMaxScaler function.

# The model

This is a WGAN-GP that we have implemented under keras. To do this, we had to customise everything that went into the keras fit function (https://keras.io/guides/customizing_what_happens_in_eit/). Much of the work was already done by [4] but the code had a few small errors. In particular, we added several options:
- callback to recover the best model during training. For this, we base ourselves on the value of the discrimination loss. Indeed, in the ideal since one uses a WGAN-GP it should converge. (see article [6]).
- Define a test_step to be able to use a set validation.
- Trackers to be able to follow the evolution of the losses and the measured metric.

## Evaluation
There are many ways to evaluate the prediction of a model in finance, the RMSE is not enough. We therefore implemented the following metrics: RMSE, R2, MAPE, POCID and SLG. \
In order to be able to easily evaluate the results of our WGAN-GP we used other more conventional models for the prediction of time series. Including Arima and a single LSTM.
All the results obtained are present in the endToEnd.py notebook, but overall what we need to remember from our experiments is that the GAN did not show much better results than the other models. \ 
However, it is a complicated model to set up and especially to finetune. So it can sometimes be difficult to use it properly. We have therefore made all the code available to you so that you can do your own tests. Indeed, many things are still being tested, such as the structure of the generator and the discriminator or the types of data that are really useful for input. But never forget that sometimes it is better to make it simple.

## Bibliography
-	[1] Using the latest advancements in deep learning to predict stock price movements  
Boris b \
https://towardsdatascience.com/aifortrading-2edd6fac689d

-	 [2] GENERATIVE ADVERSARIAL NETWORKS IN TIME SERIES: A SURVEY AND TAXONOMY \
 Eoin Brophy, Zhengwei Wang, Qi She, Tomás Ward https://arxiv.org/pdf/2107.11098.pdf

-	[3] Multi-Model Generative Adversarial Network Hybrid Prediction Algorithm (MMGAN-HPA) for stock market prices prediction
Author links open overlay panel\
 Subba RaoPolamuriaDr. ,KudipudiSrinivasbDr. ,A.Krishna Mohanc https://www.sciencedirect.com/science/article/pii/S1319157821001683

-	[4] Stock price prediction using Generative Adversarial Networks \
HungChun Lin1, Chen Chen1, GaoFeng Huang1 and Amir Jafari2 https://thescipub.com/abstract/jcssp.2021.188.196

-	 [5] “Generative adversarial nets.” Advances in neural information processing systems.    Goodfellow, Ian, et al.


-	[6] Improved Training of Wasserstein GANs  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville 
https://arxiv.org/abs/1704.00028 

-	[7] Finbert implementation
https://github.com/ProsusAI/finBERT.
