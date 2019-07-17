# NaiveBayes Text Classification

Python implementation of a Naive Bayes classifier which takes a series of text documents and categorizes them into five different categories: business, entertainment, sport, politics and tech by applyng multi-category classification.

# File description

*main.py* : the python implementation of the algorithm\
*stopwords-en.txt* : a text file containing the main english stopwords

## Build the project

Once you have downloaded the repository, you have to create a folder named "Dataset" inside the repository.

Then you can download the dataset here: http://mlg.ucd.ie/datasets/bbc.html by clicking on " >> Download raw text files ".

The download will produce a folder named "bbc" containing the five folders (one for each category) that you will have to move inside the "Dataset" folder created before.

As there is a different number of documents for each category, I chose to equalize them to the minor number (386).

At this point the structure of your project should be the following:

NaiveBayes-Text-Classification-master
 - main.py
 - stopwords-en.txt
 - Dataset
   - business (containing business text files)
   - entertainment (containing entertainment text files)
   - politics (containing politics text files)
   - sport (containing sport text files)
   - tech (containing tech text files)

## Run the project

Run the project is trivial, you must use the following command line inside the root directory:

```
python3 main.py

```



