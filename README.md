# DSI Resource

This repo contains resources for DSI learners, split by various sections below.

---

### Section 1A - Python Programming

#### Online Resources
* [Coursera - Google IT Automation with Python (Professional Certificate)](https://www.coursera.org/professional-certificates/google-it-automation)  
Python course offered by Google on Coursera consisting of 4 courses, and includes an introduction to Git and Github. 

* [Real Python](https://realpython.com/)  
Great resource for Python with extensive eplanations and code examples.  
Find topics on [Errors and Exception Handling](https://realpython.com/python-exceptions/), [Lambda Functions](https://realpython.com/python-lambda/), 
[Decorators](https://realpython.com/primer-on-python-decorators/), [Object-Oriented Programming ](https://realpython.com/python3-object-oriented-programming/) etc.

* [Think Python, 2nd Edition](https://greenteapress.com/wp/think-python-2e/)  
Beginner book on Python covering a wide range of basic Python. Suitable for beginners to programming.  
Allen Downey writes several books using Python. Check them out [here](https://greenteapress.com/wp/). Special mention of [Elements of Data Science](https://allendowney.github.io/ElementsOfDataScience/README.html), which is targetted at newbies to Python and Data Science. Good design and navigation.

---

### Section 1B - Probability and Statistics

#### Online textbooks
* [Introductory Statistics](https://openstax.org/books/introductory-statistics/pages/1-introduction)  
Good review of Statistics covering descriptive statistics, confidence intervals, hypothesis-testing etc.  
Clear explanation of Statistic terms. No Python code examples.

* [Think Stats2](http://allendowney.github.io/ThinkStats2/)  
Another great resource by Allen Downey. Use of some custom packages, but otherwise, good resource.   

* [Introduction to Probability for Data Science](https://probability4datascience.com/index.html)  
In-depth study of Probability and Statistics. Contains concise review of required mathemtics (eg. set theory, linear algebra).  
Contains measure-theoretic perspective when introducing Probability.   
Contains videos (ie. not all topics), Python code examples, good graphics.  


#### Videos
* Statquest
Great short videos explaining complex ideas.
  * [Main Ideas behind Probability Distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc)
  * [Binomial Distribution and Test](https://www.youtube.com/watch?v=J8jNoF-K8E8)
  * [Sampling from a Distribution](https://www.youtube.com/watch?v=XLCWeSVzHUU)
  * [The Central Limit Theorem](https://www.youtube.com/watch?v=YAlJCEDH2uY&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9&index=20)
  * [Bootstrapping](https://www.youtube.com/watch?v=Xz0x-8-cgaQ&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9&index=24)
  * [Confidence Intervals](https://www.youtube.com/watch?v=TqOeMYtOc1w&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9&index=23)
  * [Hypothesis Testing with SciPy - Hillary Green-Lerman](https://www.youtube.com/watch?v=dPXBN8ms-cU) - Great hands-on tutorial on the main hypothesis tests!


#### Additional Resources
* Confidence Interval - [Margin of Errors](https://web.stat.tamu.edu/~suhasini/teaching301/stat301MoE.pdf), [The t-distribution](https://web.stat.tamu.edu/~suhasini/teaching301/stat301CI_t-dist.pdf)  
From Texas A&M University. Some useful notes
* [Interpreting Confidence Interval](https://rpsychologist.com/d3/ci/)  
Insightful visualization

---

### Section 2A - Working with Data with Pandas etc.


Visualization/Interactive tools include 
[Matplotlib](https://matplotlib.org/), 
[Seaborn](https://seaborn.pydata.org/), 
[ipywidgets](https://ipywidgets.readthedocs.io/en/stable/), 
[Plotly](https://plotly.com/), 
[nbinteract](https://www.nbinteract.com/), 
[Streamlit](https://streamlit.io/) etc.

#### Online Resources
* [Chris Albon](https://chrisalbon.com/)  
Great resource by Chris Albon (has EVERYTHING!). [Joining and merging Pandas dataframe](https://chrisalbon.com/code/python/data_wrangling/pandas_join_merge_dataframe/) is a favourite! 

* [Matplotlib](https://matplotlib.org/)  
Matplotlib website contains examples, tutorials, cheat sheets, documentation and references for all things Matplotlib.

* [seaborn](https://seaborn.pydata.org/)  
Seaborn is another popular visualization tool. Site contains examples, Tutorial and references.

---

### Section 3 - Linear Regression



#### Online Resources to understand OLS and R2
* [OLS Regression](https://setosa.io/ev/ordinary-least-squares-regression/) - Visual and Interactive
* [Mean Square Error & R2 Score Clearly Explained](https://www.bmc.com/blogs/mean-squared-error-r2-and-variance-in-regression-analysis/)
* [Linear Regression Viz - Residual Sum of Squares](https://www.geogebra.org/m/UxJQorBl)
* [The Game of Increasing R-squared in a Regression Model](https://www.analyticsvidhya.com/blog/2021/05/the-game-of-increasing-r-squared-in-a-regression-model/)
* [Should we calculate R2 for training and test data?](https://stats.stackexchange.com/questions/348330/should-r2-be-calculated-on-training-data-or-test-data) - good discussion. Calculate R2 for training data to see how well the model is fit. Revert to MSE or RMSE when comparing training and test data for generalisation.


#### Online Resources to understand Bias-Variance Tradeoff
* [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) - by Statquest
* [Bias Variance Trade-off Easily Explained | Machine Learning Basics](https://www.youtube.com/watch?v=1JWpXHgqj54) - by Professor Ryan. I like the detail explanation. I just felt that instead of Sum of Squared Errors, he used Mean Squared Errors when comparing bias.
* [How to Calculate the Bias-Variance Trade-off with Python](https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/) - An example that uses the mlxtend package to breakdown the MSE into bias and variance components.

#### Multicollinearity
* [Detect and deal with Multicollinearity](https://towardsdatascience.com/how-to-detect-and-deal-with-multicollinearity-9e02b18695f1) - good explanation
* [Detect Multicollinearity using VIF](https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/) - simple example
* [Permutation Importance with Multicollinearity](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html) - considerations for feature importance


#### Online Resources to understand Cross-Validation
* [Scikit-Learn - Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html) - A useful resources to understand how cross-validation works in scikit-learn, and how the cross_val_score function is used, and how it differs from cross_val_predict.
* [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) - by Statquest

#### Online Resources to understand Regularisation
* [Regularization - Ridge](https://www.youtube.com/watch?v=Q81RR3yKn30), [Regularization - Lasso](https://www.youtube.com/watch?v=NGf0voTMlcs), [Regularization - Elastic Net](https://www.youtube.com/watch?v=1dKRdX9bfIo) - Statquest has a series to explain the various regularization techniques used. Watch in that order.

### Online Resources to understand statsmodel summary
* [Explaining Statsmodel Summary](https://towardsdatascience.com/simple-explanation-of-statsmodel-linear-regression-model-summary-35961919868b) - a more detailed explanation with recommended range of values.
* [Interpreting Linear Regression through Statsmodel Summary](https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a) - quick overview

### GridSearchCV multiple models, multiple parameters
* [Tune multiple models with CV all at once](https://towardsdatascience.com/how-to-tune-multiple-ml-models-with-gridsearchcv-at-once-9fcebfcc6c23)


---

### SQL

#### Online Resources for SQL
* [SQLBolt](https://sqlbolt.com/) - Simple, interactive exercises
* [SQL Zoo](https://sqlzoo.net/wiki/SQL_Tutorial) - Interactive exercises that contains intermediate queries for Windowing, and COVID example
* [W3 Schools](https://www.w3schools.com/sql/default.asp) - Good reference and code examples.
* [SQL Views](https://www.w3schools.com/sql/sql_view.asp) - Always updated.
* [SQL Server - Materialized Views](https://docs.microsoft.com/en-us/sql/t-sql/statements/create-materialized-view-as-select-transact-sql?view=azure-sqldw-latest) - Microsoft SQL Server specific Materializeed Views

---

### Feature Engineering

#### Online Resources for Feature Selection

* [Recursive FEature Elimination](https://machinelearningmastery.com/rfe-feature-selection-in-python/) - An example
* [SelectKBest](https://www.datatechnotes.com/2021/02/seleckbest-feature-selection-example-in-python.html)

#### Outliers
* [Interquartile Range Rule](https://www.thoughtco.com/what-is-the-interquartile-range-rule-3126244) - IRQ Rule to detect outliers


---

### Classification

### Online Resources for Logistic Regression
* [Loss Function for Logistic Regression](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11) - detailed and code exasmples.


#### Online Resources for Multiclass problems
* [Machine Learning Mastery - 1-vs-rest, 1-vs-1 Multiclass](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/) - good explanation and code examples for multiclass problem
* [LinearSVC Multiclass example](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)
* [Scikit-Learn LinearSVC document that shows 1-vs-rest implementation for multiclass problems](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

#### Online Resources for Imbalance Data
* [Intro to imbalanced data classification](https://machinelearningmastery.com/what-is-imbalanced-classification/)
* [SMOTE for imbalanced data](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)

#### Online Resources for Classification Model Evaluation

* [Evaluating a Classification Model](https://www.ritchieng.com/machine-learning-evaluate-classification-model/) - Many steps are involved, but quite comprehensive

* [Evaluate the Business Value of a Machine Learning Model](https://www.analyticsvidhya.com/blog/2021/09/how-to-evaluate-the-business-value-of-ml-model/) - Focus on Decile plots and lift plots

---

### Web Dev Resources

#### Online Resources for Web Dev
* [W3School - HTML tutorial](https://www.w3schools.com/html/) - HTML, CSS and other examples
* [W3C - CSS Properties](https://www.w3.org/wiki/CSS/Properties/color/keywords) - W3C standards
* [Codepen](https://codepen.io/) - Live coding environment.
* [Microsoft - Web Dev for Beginners](https://github.com/microsoft/Web-Dev-For-Beginners) - Project-based approached to web development.

---

### Regex

* [Regex 101](https://regex101.com/) - Interactive website to practice Regex
* [Regex Crossword](https://regexcrossword.com/) - Learn regex using crosswords
* [Regexone](https://regexone.com/) - Tutorial-like step-by-step lesson for beginners
* [Regex Golf](https://alf.nu/RegexGolf?world=regex&level=r00) - For practice

---

### NLP

* [VADER](https://medium.com/mlearning-ai/vader-valence-aware-dictionary-and-sentiment-reasoner-sentiment-analysis-28251536698) - VADER Example for Sentiment Analysis
* [Embedding Projector](https://projector.tensorflow.org/) - Great Interactive Visualization for Word2Vec
* [Video on Word2Vec - Skipgram and CBOW](https://www.youtube.com/watch?v=UqRCEmrv1gQ) - Deeper understanding on the 2 methods for Word2Vec
* [Pix2Story - Generate stories from pictures](https://azure.microsoft.com/en-in/blog/pix2story-neural-storyteller-which-creates-machine-generated-story-in-several-literature-genre/)
* [Gensim code example, with Viz using PCA](https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html)
* [Word Analogies (ie. Word Vectors)](https://kawine.github.io/blog/nlp/2019/06/21/word-analogies.html)
* [Pipeline example - Countvectorizer, TFIDF-Transformer, SGDClassifier](https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html)
* [Text Classification - FeatureUnion](https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html) - Complex pipeline example
* [Text Classification Example](https://appliedsingularity.com/2022/02/01/nlp-tutorials-part-6-text-classification/) - Contains GLOVE/word embedding example in the later part.

---

### Naive Bayes

* [Statquest Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) - Watch the Gaussian NB after this.
* [Statquest Gaussian NB](https://www.youtube.com/watch?v=H3EjCKtlVog)
* [How NB Works](https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/) - Example calculations, Laplace Correction (for missing value)
* [NB Algorithm](https://www.analyticsvidhya.com/blog/2021/09/naive-bayes-algorithm-a-complete-guide-for-data-science-enthusiasts/) - Another website with worked examples
* [NB Classifier - Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) - Good example on Gaussian NB
* [NB Example](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)

---

### SVM

* [Statquest SVM](https://www.youtube.com/watch?v=efR1C6CvhmE) - First of 3 videos.
* [Kernel Trick](https://drewwilimitis.github.io/The-Kernel-Trick/) - Nice graphs to demo kernel trick
* [Kernel trick](https://medium.com/@zxr.nju/what-is-the-kernel-trick-why-is-it-important-98a98db0961d) - actual calculations for polynomial degree to to see reduction in calculations

---

### Classification Metrics

* [AUC-ROC example cdoe](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/)

---

### GLM

* [CAS Monograph on GLM for Insurance](https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf)

* [Deviance Statistics](https://www.clayford.net/statistics/comparing-multilevel-models-using-deviance-statistics-ch-4-of-alda/#:~:text=To%20use%20the%20Deviance%20Statistic,bigger%20model%20equal%20to%200.&text=Smaller%20Deviance%20is%20better.,may%20be%20a%20good%20thing.) - only suited for nested model comparisons


---

### Ensemble


* [scikit-learn Ensemble Methods user guide](https://scikit-learn.org/stable/modules/ensemble.html#stacking) - take note of VotingClassifier and VotingRegressor to implement stacking.
* [Hyperparameter tuning for VotingClassifier, GridsearchCV](https://stackoverflow.com/questions/46580199/hyperparameter-in-voting-classifier)
* [Statquest - Adaboost](https://www.youtube.com/watch?v=LsK-xG1cLYA)
* [Debiasing RF for Treatment Effect Estimation](https://www.youtube.com/watch?v=I3GNxTjJPWk)
* [Statquest - Gradient Boost](https://www.youtube.com/watch?v=3CC4N4z3GJc) - Part 1 of 4
* [Hyperparameter tuning of Random Forest Classifier](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74) - RandomSearchCV

---

### Feature Importance

* [Scikit-Learn example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
* [Feature Importance Example](https://mljar.com/blog/feature-importance-in-random-forest/)
* [scikit-learn permutation importance example](https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance)
* [Video explaining Permutation Importance](https://www.youtube.com/watch?v=VUvShOEFdQo) - Sebastian Raschka

---

### Gradient Descent

* [Video on Partial Differentiation and finding inflexion points](https://www.youtube.com/watch?v=mMCsFbAtDjk)
* [Statquest -  Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8)

---

### Clustering

* [RealPython - K-Means Clustering](https://realpython.com/k-means-clustering-python/) - Good examples with elbow method and silhoutte method.
* [K-means Clustering - interactive visualization](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
* [Mail Customers Segmentations with k-Means](https://medium.com/@budisumandra/mall-customers-segmentation-with-k-means-clustering-algorithm-in-python-55ba10e4bbe3) - Business insights from post-clustering analysis
* [Exploring Customers Segmentation With RFM Analysis and K-Means Clustering](https://medium.com/web-mining-is688-spring-2021/exploring-customers-segmentation-with-rfm-analysis-and-k-means-clustering-118f9ffcd9f0)
* [K-means - great write-up and Viz](https://www.pinecone.io/learn/k-means-clustering/)

---

### PCA

* [PCA](https://setosa.io/ev/principal-component-analysis/) - Interactive Viz
* [PCA - Statquest](https://www.youtube.com/watch?v=FgakZw6K1QQ)


### Time Series

* [Forecasting: Principles and Practice Book - Time Series Components](https://otexts.com/fpp2/components.html) - Book on Forecasting and time series componants as example
* [Times Series Decomposition](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/) - Additive and Multiplicative Example
* [Aric LaBarr - Time Series Videos](https://www.youtube.com/c/AricLaBarr/videos) - Great collection of short videos on Times Series. Start off with   [What is Time Series Data](https://www.youtube.com/watch?v=FsroWpkUuYI), then [What is Time Series Decomposition](https://www.youtube.com/watch?v=0ar9extHObg), then [What are AR Models](https://www.youtube.com/watch?v=Mc6sBAUdDP4), then [What are MA models](https://www.youtube.com/watch?v=zNLG8tsA_Go), then [What are ARIMA models](https://www.youtube.com/watch?v=dXND1OEBABI). 
* [Statsmodel - SARIMAX documentation](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html)
* [SARIMAX example](https://analyticsindiamag.com/complete-guide-to-sarimax-in-python-for-time-series-modeling/#:~:text=SARIMAX-,SARIMAX(Seasonal%20Auto%2DRegressive%20Integrated%20Moving%20Average%20with%20eXogenous%20factors,average%20component%20in%20the%20model.) - good and simple example. End-to-end. 

* [Intro to ACF, PACF](https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/) - ML Mastery
* [PACF - PennState](https://online.stat.psu.edu/stat510/lesson/2/2.2)
* [Identify order of ARIMA models - ACF/PACF](https://www.youtube.com/watch?v=ZE_WGBe0_VU&ab_channel=LasseEngboChristiansen)


---

### Graph/Network Models

* [Enron Email](http://hci.stanford.edu/~jheer/projects/enron/v1/) - Great visualizations.

---

### Spatial Regression

* [Spatial Regression](https://geographicdata.science/book/notebooks/11_regression.html) - Part of an online book, Geographic Data Science with Python
* [Spatial Regression](http://darribas.org/gds_scipy16/ipynb_md/08_spatial_regression.html) - Geographic DS with PySAL library.


---

### Bayes Modelling
- [Bayes Rules!](https://www.bayesrulesbook.com/) - Introductory online book on Bayes Modelling
- [Bayesian Inference Framework - Video](https://www.youtube.com/watch?v=0w_4QcvBYII&ab_channel=MITOpenCourseWare)
- [Decision Analysis - Posterior Probability Calcs](https://www.youtube.com/watch?v=FpKiHpYnY_I&ab_channel=JoshuaEmmanuel)
- [Bayesian Stats Part 1 Video](https://www.youtube.com/watch?v=0F0QoMCSKJ4&ab_channel=StataCorpLLC)
- [Bayesian Stats Part 2 Video](https://www.youtube.com/watch?v=OTO1DygELpY&ab_channel=StataCorpLLC)

### NN and Deep Learning

* [FastAI Practical Deep Learning Course](https://course.fast.ai/)
* [Transformer Models - intro and catalog for 2022](https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/)
* [NN Loss Visualization](https://www.telesens.co/2019/01/16/neural-network-loss-visualization/)
* [Batch Sizing](https://stackoverflow.com/questions/63139072/batch-size-for-stochastic-gradient-descent-is-length-of-training-data-and-not-1#:~:text=batch_size%20is%20the%20size%20of,gradient%20from%20your%20entire%20dataset.) - Some queries and answer
* [Intro to Mini-batch Grad Descent and how to configure Batch Size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)

* [NN Playground!](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.65170&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) - Google's tensorflow NN playground
* [Video - Visual Intro to Neural Networks, Chap 1](https://www.youtube.com/watch?v=UOvPeC8WOt8). Chap 2 is also quite good!

---

### Computer Vision, Image Processing

* [Object Detection YOLO v7](https://www.analyticsvidhya.com/blog/2022/08/how-to-train-a-custom-object-detection-model-with-yolov7/)
* [Object Detection YOLO v5](https://www.analyticsvidhya.com/blog/2021/12/how-to-use-yolo-v5-object-detection-algorithm-for-custom-object-detection-an-example-use-case/)
* [Semantic Segmentation](https://towardsdatascience.com/building-a-semantics-segmentation-computer-vision-algorithm-for-deployment-on-the-edge-3ad1a8922fd1)

---

### Pipelines and Model Deployment

* [Sklearn Pipelines Example](https://medium.com/analytics-vidhya/machine-learning-models-to-production-part-1-build-your-own-sklearn-pipeline-e7aa7c06152a)
* [Pipelines with MLflow](https://towardsdatascience.com/machine-learning-model-development-and-deployment-with-mlflow-and-scikit-learn-pipelines-f658c39e4d58)


---

### Transfer Learning/Pre-trained Models
* [Tensorflow Hub](https://www.tensorflow.org/hub)


---

### Cloud

* [Microsoft Azure - Free training and exam for cloud/data/AI fundamentals](https://www.microsoft.com/en-sg/apac/digitalevent)

---

### Websites with Data

* [US Govt Data](https://www.data.gov/)
* [US Census Data](https://www.census.gov/data.html)
* [CIA World Factbook](https://www.cia.gov/the-world-factbook/)
* [US Health Data](https://healthdata.gov/)
* [AWS Open Data](https://registry.opendata.aws/)
* [Google Dataset Search](https://datasetsearch.research.google.com/)
* [UCI ML Repo](http://archive.ics.uci.edu/ml/index.php)
* [Data World](https://data.world/)
* [Perma CC time series data](https://perma.cc/56Q5-YPNT)
* [Data Portal](https://dataportals.org/)
* [Kaggle](https://www.kaggle.com/)

---

### Data Science Resources

#### Book

* [Machine Learning, Tom Mitchell ](http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/mlbook.html)

* [An Introduction to Statistical Learning](https://www.statlearning.com/)
Free. Great introduction to DS. Examples in R. Now in it's 2nd edition.

* [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)
Free to download. 2nd edition. Examples in R. More advanced book.

* [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/w~shais/UnderstandingMachineLearning/index.html) - Free

* [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf)

* [Deep Learning Book](https://www.deeplearningbook.org/)
Website has exercises and lecture notes. 

- [Python for Data Analysis, 3E](https://wesmckinney.com/book/) - Classic Pandas book by the originator, Wes McKinney. Updated for Pandas 1.4, Python 3.10.

- [Machine Learning Mastery](https://machinelearningmastery.com/) - Great resources!

- [Chris Albon](https://chrisalbon.com/) - Another great resource website for data science, pandas etc. 



---

### Git

#### Online Resources
* [An Introduction to Git and Github](https://www.youtube.com/watch?v=MJUJ4wbFm_A) by Brian Yu  
This introductory video by Brian provides an overview to Git and how it's used.

* [Create Github Profile - Video](https://www.youtube.com/watch?v=Y1z7_GfEPiE)


### Todo

* jupyter notebook
* Data Science - Read Intro to Probability for DS Chap 6, deriving Hoeffding's Inequality. Then continue with 'Learning from Data', with slight changes to the formula to relate to generalization error.