from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import json, pandas
from nltk.stem.snowball import SnowballStemmer
import random
import numpy as np



def getTrainingData(filename):
    # get training data
    training_dicts = []
    for line in open(filename,'r'):
        training_dicts.append(json.loads(line))

    training_headlines = []
    training_categories = []
    for dct in training_dicts:
        training_headlines.append(dct["headline"])
        training_categories.append(dct["category"])

    return training_headlines, training_categories


def getTestData(filename):
    # get test data
    df = pandas.read_csv(filename)
    test_headlines = list(df['Title transcription'].values.astype(str))
    new_test_headlines = []
    for headline in test_headlines:
        new_headline = headline.replace("‚Äô", "'")
        new_headline = new_headline.replace("‚Äò", "'")
        new_headline = new_headline.replace("‚Äî", ",")


        new_test_headlines.append(new_headline) #apostrophes and commas in csv file need to be changed

    return new_test_headlines



def createSklearnModel(training_headlines, training_categories):

    stemmer = SnowballStemmer("english", ignore_stopwords=True) #stemming
    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    #create model with a pipeline
    text_clf_svm = Pipeline([('vect', StemmedCountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(early_stopping=True, random_state=42))])
    text_clf_svm = text_clf_svm.fit(training_headlines, training_categories)

    parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

    #Use gridsearch
    gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
    gs_clf_svm = gs_clf_svm.fit(training_headlines, training_categories)

    return gs_clf_svm


def writeToCSV(filename, predicted_categories):
    df = pandas.read_csv(filename)
    df['Category'] = predicted_categories
    print(df)
    df.to_csv(filename)


def trainingAccuracyTest(training_headlines, training_categories):
    #list of unique categories
    unique_categories = list(set(training_categories))

    #shuffle randomly
    shuffled = list(zip(training_headlines, training_categories))
    random.shuffle(shuffled)
    headlines, categories = zip(*shuffled)

    #split data
    training_head = headlines[:199600]
    training_cat = categories[:199600]
    test_head = headlines[199600:]
    test_cat = categories[199600:]

    #create model and predict
    model = createSklearnModel(training_head, training_cat)
    predicted_categories = model.predict(test_head)

    #performance metrics
    report = classification_report(test_cat, predicted_categories, output_dict=True, target_names=unique_categories)
    df = pandas.DataFrame(report).transpose()
    df.to_csv('/Users/howardqian/Desktop/ML and CNN/corsali-machine-learning-technical-interview-main/performance_metrics.csv')
    print(df)

    #confusion matrix
    unique_label = np.unique([test_cat, predicted_categories])
    cmtx = pandas.DataFrame(
        confusion_matrix(test_cat, predicted_categories, labels=unique_label),
        index=['true:{:}'.format(x) for x in unique_label],
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    cmtx.to_csv('/Users/howardqian/Desktop/ML and CNN/corsali-machine-learning-technical-interview-main/confusion_matrix.csv')
    print(cmtx)













