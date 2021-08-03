from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import json, pandas
from nltk.stem.snowball import SnowballStemmer




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

