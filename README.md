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

#### Online Resources to understand Bias-Variance Tradeoff
* [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) - by Statquest
* [Bias Variance Trade-off Easily Explained | Machine Learning Basics](https://www.youtube.com/watch?v=1JWpXHgqj54) - by Professor Ryan. I like the detail explanation. I just felt that instead of Sum of Squared Errors, he used Mean Squared Errors when comparing bias.
* [How to Calculate the Bias-Variance Trade-off with Python](https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/) - An example that uses the mlxtend package to breakdown the MSE into bias and variance components.


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


---

### Data Science Resources

#### Book
* [An Introduction to Statistical Learning](https://www.statlearning.com/)
Free. Great introduction to DS. Examples in R. Now in it's 2nd edition.

* [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)
Free to download. 2nd edition. Examples in R. More advanced book.

* [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/w~shais/UnderstandingMachineLearning/index.html) - Free

* [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf)

* [Deep Learning Book](https://www.deeplearningbook.org/)
Website has exercises and lecture notes. 

- Chris Albon
- Machine Learning Mastery
- 



---

### Git

#### Online Resources
* [An Introduction to Git and Github](https://www.youtube.com/watch?v=MJUJ4wbFm_A) by Brian Yu  
This introductory video by Brian provides an overview to Git and how it's used.



### Todo

* jupyter notebook
* Data Science - Read Intro to Probability for DS Chap 6, deriving Hoeffding's Inequality. Then continue with 'Learning from Data', with slight changes to the formula to relate to generalization error.