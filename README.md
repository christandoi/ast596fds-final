# ast596fds-final
Final group project for Frank Fu, Sihan Li, Jennifer Li, Chris Tandoi

# Project proposal:
Astrophysical dataset that we find interesting. Project must satisfy (at least) two of:

	- time-series analysis
    - hierarchical Bayesian modeling
    - dealing with selection effects in data when building models
    - machine learning	

------------------------
Sunspot modeling and prediction based on historical data located at 
- https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-indices/sunspot-numbers/american/tables/
- http://www.sidc.be/silso/datafiles

Possible scientific questions:
1) Characterize the variability of sunspot numbers using different statistical tools
2) Make a model for prediction
3) Discuss how significance the weakening trend is (or other trends in the variability)

Also interesting to add to our final writeup: history of sunspot detection, dating back from ~800BC in China (Book of Changes) to first telescopically observed sunspots in the early 1600's

Some reference images:

![SIDC daily sunspot number since 1900](images/solar_cycles_since_1900.PNG)

# Statement of work:
How do we want to accomplish this? Think about final outcome, work backwards on how to get there. What is needed at each step? Work will be done in Jupyter notebook

------------------------

Methods:
1) PCA analysis on different timescales, show some analysis on counting uncertainties
2) For the predicting model: something similar to previous homeworks
3) Use MCMC with the model to quantify significance of certain trends


# Tasks:

### Frank Fu
- Fourier Component reconstruction (nb2)
- Gaussian Process regression (nb1)

### Sihan Li
- Regression and find the period
- Machine Learning (Comparison and analysis)

### Jennifer Li

### Chris Tandoi
- Modeling + MCMC
- Writeup some interesting history to tantalize the audience
