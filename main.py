from prediction import content_classification as cc



def main():
    training_filename = '/Users/howardqian/Desktop/ML and CNN/corsali-machine-learning-technical-interview-main/data/news_dataset.json'
    test_filename = '/Users/howardqian/Desktop/ML and CNN/corsali-machine-learning-technical-interview-main/data/post_data.csv'

    #get training data
    training_headlines, training_categories = cc.getTrainingData(training_filename)
    #get test data
    test_headlines = cc.getTestData(test_filename)

    #create model
    model = cc.createSklearnModel(training_headlines, training_categories)
    #make test data predictions using model
    predicted_categories = model.predict(test_headlines)

    for i in range(len(predicted_categories)):
        #many test posts do not have a title
        if test_headlines[i] == "None" or test_headlines[i] == "No title" or test_headlines[i] == "No Title" \
                or test_headlines[i] == "Unavailable" or test_headlines[i] == "No title." or test_headlines[i] == "none":
            predicted_categories[i] = "NO CATEGORY"

        print(predicted_categories[i], test_headlines[i])



if __name__ == "__main__":
    main()