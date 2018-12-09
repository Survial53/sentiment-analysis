import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import preprocessing_tools as prep
import pandas as pd
import numpy as np
import datetime

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'bag_of_words_meets_bags_of_popcorn', 'labeledTrainData.tsv'), header=0, \
                    delimiter="\t", quoting=3)

test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'bag_of_words_meets_bags_of_popcorn', 'testData.tsv'), header=0, delimiter="\t", \
                   quoting=3 )

clean_train_reviews = []
 
# loop over each text review; create an index i that goes from 0 to the length
# of the movie review list
for i in range( 0, len(train["review"])):
    clean_train_reviews.append(" ".join(prep.review_to_wordlist(train["review"][i], True)))
    
# initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

np.asarray(train_data_features)


# initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# fit the forest
forest = forest.fit( train_data_features, train["sentiment"] )

# TEST ============================================================= 
clean_test_reviews = []

for i in range(0,len(test["review"])):
    clean_test_reviews.append(" ".join(prep.review_to_wordlist(test["review"][i], True)))

# get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
np.asarray(test_data_features)

result = forest.predict(test_data_features)

# copy the results to a pandas dataframe with "id" column and
# "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

#now = datetime.datetime.now()

#out_name = "Bag_of_Words_model_" + now.strftime("%Y-%m-%d %H:%M") + ".csv"

output.to_csv(os.path.join(os.path.dirname(__file__), 'result', 'Bag_of_Words_model.csv'), index=False, quoting=3)