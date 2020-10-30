# COVID19 Topic-Modeling

![Screenshot 2020-10-28 at 19 36 55](https://user-images.githubusercontent.com/11573356/97481416-16c73280-1955-11eb-8740-614883415b62.png)

COVID19 Topic-Modeling is an NLP task meant to help reveal latent topics found in a collection of documents from Covid-19 Open Research Dataset.
The aformentioned dataset can be found at the following url: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7251955/

It serves as a study case in Topic Modeling using state-of-the-art methods: LDA, LDA_MALLET, LSI, HDP, NMF 

Technologies used: Python3, Gensim, Spacy, pyLDAVis, Jupyter Notebook, matplotlib

## Evaluation:

  |      Method   |     c_v       | u_mass    | alpha | eta |
  | ------------- |:-------------:| ---------:|------:|----:|
  | **LDA**       | **0.546**     | -2.790    | 0.1  | 0.1|
  | LDA_MALLET    | 0.517         | -2.038    | 0.1  | 0.1|
  | LSI           | 0.349         | -2.115    | 0.1  | 0.1|
  | HDP           | 0.364         | -1.893    | 0.1  | 0.1|
  | NFM           | 0.513         | -2.563    | 0.1  | 0.1|
  
## Project Outline:
```
  - Generating ground-truth dataset
  - Model Training & Evaluation
  - Topic Prediction

Basic project installation steps:

  1. Clone repository

  2. Generate model & evaluation files:
     - preprocess list of documents
     - generate list of document tokens 
     - import and create Evaluation object
     - create model using create_model() function
     - save model & evaluation files to a given output path
     
     Sample:
          from evaluation import Evaluation
          
          token_lists = [
                          ["virus", "outbreak", "virus", "pandemic"], 
                          ["doctor", "risk_factor", "health", "health"], 
                          ["coronavirus", "covid", "hospital", "healthcare"], 
                          ["death", "respiratory_problem", "factor", "people"]
                        ]
          
          ev = Evaluation(lang_code="en", method="LDA", version="1.1", k=50, alpha=0.1, eta=0.1, num_words=15)	
          ev.create_model(token_lists, output_path=output_path)	
   
          ev = Evaluation(lang_code="en", method="LDA_MALLET", version="1.1", k=50,  alpha=0.1, eta=0.1, num_words=15)	
          ev.create_model(token_lists, output_path=output_path)	
   
          ev = Evaluation(lang_code="en", method="LSI", version="1.1", k=50,  alpha=0.1, eta=0.1, num_words=15)	
          ev.create_model(token_lists, output_path=output_path)
          
          ev = Evaluation(lang_code="en", method="HDP", version="1.1", k=50, alpha=0.1, eta=0.1, num_words=15)	
  
          ev = Evaluation(lang_code="en", method="NMF", version="1.1", k=50, alpha=0.1, eta=0.1, num_words=15)	
          ev.create_model(token_lists, output_path=output_path)	
    
     Evaluation files:
        - plot pyLDAVis topic distribution
        - topic wordclouds
        - evaluation metrics (c_v, u_mass, perplexity)
        - topic_terms 
  
  3. Predict topic for new documents:
      - import and create Topic object
      - predict topic using predict_topic() function

   Sample:
         from topic import Topic
         text = "In December 2019, a novel coronavirus called SARS-CoV-2 has resulted in the outbreak of a respiratory illness known as COVID-19"
         t = Topic(lang_code="en", method="LDA", version="1.1", k=50, top_k=1, num_words=5)
         pred = t.predict_topic(text)
         print(pred)
         
         '''    
               {
               "predictions":[
                  {
                     "id":12,
                     "confidence":"0.209",
                     "topic_terms":[
                        {
                           "term":"virus",
                           "weight":"0.137"
                        },
                        {
                           "term":"coronavirus",
                           "weight":"0.029"
                        },
                        {
                           "term":"spread",
                           "weight":"0.027"
                        },
                        {
                           "term":"outbreak",
                           "weight":"0.023"
                        },
     
                        {
                           "term":"infection",
                           "weight":"0.018"
                        }
                     ]
                  }
               ],
               "message":"successful"
            }
        '''
       
```
