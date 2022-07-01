# Medium Article Analysis and Sucess Predictor

## Overview

This project is an investigative approach in to the determinants of an article success in Medium. It was completed as part of my capstone project for the Data Science Immersive course at Gerneal Assembly. The topic was chosen as Medium has been an incredible friend and ally throught this intense course and I wanted to do a deeper dive into it.

The code for the final deployment of the model can be found here: https://github.com/Arik-LT/medium_streamlit_webapp. It is not currently deployed online as the size of the project didn't qualify for the free tier, however it is very easy to get up and running locally.

![Alt text](assets/medium.jpeg?raw=true)

## Contents

| Notebook Name                            | Contents                                           |
| ---------------------------------------- | -------------------------------------------------- |
| Scraper.ipynb                            | Code for web scraper used to extract the data      |
| Data Wrangling.ipynb                     | Notebook for wrangling and feature engeneering     |
| EDA.ipynb                                | Exploratory Data Analysis Notebook                 |
| Modelling Global.ipynb                   | Machine Learning Models and conclusions            |
| Article recommender with TF - IDF .ipynb | Notebook used to calculate article recommendations |

## Goals and Results

Questions:

Medium has become such a widespread and used tool for knowledge in so many different fields that I thought investigating what made an article go viral or not was something quite worth exploring. Answering questions like:

- What are the characteristics features of a succesful article?
- Are there any particular topics/words that receive more attention?

Method:

- Classification task, achieved by splitting the amount of likes by median to create two classes.
- Target: Claps (aka likes)
- Median Value for claps: 95 claps

Main Models used:

- Logistic Regression
- Linear SVM
- XGBoost

Results:

- Accuracy score of 0.7, ie was able to correctly classify articles as belonging to class 0 (less than 95 claps) or class 1 (over 95 claps) with 70% accuracy
- 0.20 over Baseline

## Data Collection

To acquire the data I created a web scraper using beatifulsoup, requests and newspaper3k. Due to the paywall some articles appeared very short, capped at between 150 to 200 words so I adjusted my scraper accordingly to only incorporate articles longer than that. You can see visualy below how the process looked.

![Alt text](assets/medium.png?raw=true)

## Data Cleaning

To clean the data I took the following steps:

- Dropped rows with missing values for the story_url feature as this implied that no text was scraped.
- Dropped rows where the title was duplicated as some articles where posted more than once on different days.
- Transformed claps and followers into integers accordingly.
- Extracted the author handle from the author url using regex.
- Created a datetime object from the date and extracted the day of the week.
- Ensured all articles where in english using the langdetect package

  Finally to be able to use NLP on the text in the most efficient way I cleaned the text and title features:

- Lowercasing the text.
- Removing stop words.
- Removing punctuation and numbers.
- Lemmatized the text.

## Feature Engeneering

Features added to the dataset:

- Publication Name - Which was used as part of the for loop for the scraper and chosen by me.
- Number of words of a given text.
- Subjectivity - Extracted using the texblob package, sentiment analyser, subjectivity measures how opinionated an article is.
- Polarity - Extracted as above, polarity measures the positivity of a given text.
- Day of the week the article was published.
- I transformed the subtitle column into a binary variable where 1 means the article had a subtitle and 0 that it didn't.

## Target Variable

The target variable was the amount of claps. I created a function to classify the amount of claps as either belonging to class 0 (having less than the median amount of claps) or as belonging to class 1 (having more than the median amount).

Here we can se the histogram for claps capped at 500. The histogram is skewed to the right as it follows a downward trend. However there are two significant spikes at the 50 and 100 clap mark.

![Alt text](assets/claps_histogram_500.png?raw=true)

After some investigation it appears that most medium publications advertise an article that is doing well, once it crosses a certain threshold. This is done either by putting said article on the front page or even emailing it out to its subscribers. Our data shows that this happens at two very distinct points. That is at 50 and 100 claps.

Therefore I believe the results of this project are of even more interest to the average medium writer. Even though the scope of the project is only that of a binary classifcation, if an article manages to cross over the 100 clap mark (with are class 1 being over 95 claps) it will increase the likelyhood of it doing much better as it will get picked up by a publication and gain more exposure.

## Exploratory Data Analysis

In these two histograms we can see that this trend continues, with some publications showing much higher spikes than others.

![Alt text](assets/writing_cop_marketing.png?raw=true) | ![Alt text](assets/data_hist.png?raw=true)

The scatter plots for the added features of polarity and subjectivity also yield quite interesting insights with extremes on both sides being punished quite harshly.

![Alt text](assets/subjectivity_scatter.png?raw=true) | ![Alt text](assets/polarity_scatter.png?raw=true)

## NLP

To incorporate the use of the text and title features, I used natural language processing to split each individual and pair of words into a token. I used nltk's countvectorizer to do so, and used n_grams 1 and 2 to account for single and pairs of words.

## Modeling Overview

To be able to run the models with over 1,000,000 columns after applying count vectorizer I had to work with sparce matrices. Doing so was a bit tricky but in essence the process was to transform every column into a sparse matrix format and rejoin them together.

Finally for the remaining features I created dummy variables where needed, like day of the week and publication type.

## Models Used

- Logistic Regression
- Linear SVM
- XGBoost

_ROC COMPARISON PIC HERE_

The ROC curve measures plots the true positive rate (TPR) versus the false positive rate, as the threshold for predicting 1 changes.

If the TPR is always 1, the area under the curve is 1 (it cannot be larger). This is equivalent to perfect prediction.
When the area under the curve is 0.50, this is equivalent to the baseline (chance) prediction (marked by the diagonal line).

## Best Results

The best model was XGBoost, it achieved an accuracy score of 0.7, with baseline being 0.7. Below you can see the results of the classifcation report and the confusion matrix.

![Alt text](assets/Classification_report_best_xgb.png?raw=true) | ![Alt text](assets/confusion_matrix_best_xgb.png?raw=true)

## Coefficents

Title Coefficients:
![Alt text](assets/abs_coef_title.png?raw=true) |
Text coefficients:
![Alt text](assets/abs_coef_text.png?raw=true) |
Remaining coefficients:
![Alt text](assets/absoulte_coef_normal_features.png?raw=true) |

## Conclusions

It was very interesting that the median amount of claps was 95. As the claps histogram showed consitent spikes at the 50 and 100 clap mark. This I belive brings extra purpose and meaning to the project as belonging to class 1 (having over 95 claps) would mean a writer would be extremly close to having the medium publication advertise the article, there by compounding the total outreach and amount of interaction the article would get.

Overall the model obtained an accuracy score of 0.7 which was 0.2 points over baseline. And the streamlit interactive tool is quite a cool way to visualize these results. https://github.com/Arik-LT/medium_streamlit_webapp

## Limitations and areas of improvement

- Amount of data
  - Scrape more data for the publications with less representation
  - Focus on a single publication or topic (using LDA)
- Elastic Net for Logistic Regression - I was time constrained when doing the project with some models running over 15 hours. Adding a penalty to the logistic regression would've definitely been a good continuation of the project.
- TF-IDF
  - For whole model & more specifically for single topic approach
- Better data cleaning
  - Part of speech tagging (POS) to further reduce the amount of features
- Add amount of words in title as feature

## Key takeaways

Importance of data cleaning

Importance of EDA and of planning the workflow

## Future Work

- Network Analysis by scraping users who actually liked each article.
- Apply deep learning to the model to try to increase the predictive capability.

## Libraries used

xgboost==1.6.1
scikit-learn==1.1.1
scipy==1.8.1
requests==2.28.0
nltk==3.7
numpy==1.22.4
pandas==1.4.2
textblob==0.17.1
