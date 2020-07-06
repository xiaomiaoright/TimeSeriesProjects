# About this repo
This repo includes 5 time series projects, from entry level visualization to high level machine learning time series projects. 

# Part 1: Numpy, Pandas, and Time Series Data
Background: Pandas was developed in the context of financial modeling, so it contains a fairly extensive set of tools for working with dates, times, and time-indexed data.

## Data and time data types:
* Time stamps 2020-07-01 12:00:00
* Time intervals and periods: length of time between a particular beginning and end point
* Time deltas or durations: extact length of time

In the following section, the project will show how to work with those date types in python.
## Date and time in python
[Related Coding](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/NumpyPandasTimeSeries.ipynb)

## Time series data visualization simple example
* Data Source:  bicycle counts on [Seattle's Fremont Bridge](http://www.openstreetmap.org/#map=17/47.64813/-122.34965)
* Dataset can be download from [here](https://data.seattle.gov/Transportation/Fremont-Bridge-Hourly-Bicycle-Counts-by-Month-Octo/65db-xm6k)
* [Code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/TimeSeriesDataVisualization.ipynb)

# Part 2: Data Visualization with Pandas
In this section, a general introduction of Pandas built-in visualization is introduced. Topics:
* Different Types of Pandas visulizations
<pre>
df.plot.hist()     histogram
df.plot.bar()      bar chart
df.plot.barh()     horizontal bar chart
df.plot.line()     line chart
df.plot.area()     area chart
df.plot.scatter()  scatter plot
df.plot.box()      box plot 
df.plot.kde()      kde plot
df.plot.hexbin()   hexagonal bin plot
df.plot.pie()      pie chart
</pre>
* Two ways to call plot method:
    * df['col'].plot.hist()
    * df['col'].plot(kind = 'area')
    * df['col'].hist()
* Customizing Pandas plots
    * plotsize
    * line/marker properties:size, color
    * legend
    * plot tile, xlabel, y label
### Related Code: [here](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/PandasDataVisualization.ipynb)

# Part 3: Time Series with Pandas
## Overview
In this section, a systemtic study and project sample will be demonstrated on the functions/operations available in Pandas package to process Time Series data

## DateTime Index
* Create a date/datetime/time object in python
    * Python  Datetime overview: datetime module
    * Numpy Datetime arrays
        * The NumPy data type is called datetime64 to distinguish it from Python's datetime
        ```python
        np.array(['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64')
        # dtype: datetime64[Y],datetime64[h],datetime64[D]
        ```
    * Numpy Date Ranges
        * np.arange(start,stop,step) can be used to produce an array of evenly-spaced integers, we can pass a dtype argument to obtain an array of dates. The stop date is exclusive.
        * By omitting the step value we can obtain every value based on the precision.
        ```python
        np.arange('2018-12-31', '2019-12-31',  7, dtype='datetime64[D]')
        ```
    * Pandas Datetime Index
        * pandas.date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, **kwargs) 
        * [doc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)
        ```python
        pd.date_range('07/01/2018', periods = 7, freq='D)
        ```
        * pandas.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)
        * [doc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html)
        ```python
        pd.to_datetime(['1-2-18'], format='%d-%m-%Y')
        ```
        * create datetime index:
            * method 1: create numpy date array, then convert to pd datetime index
            ```python
            d = np.arange('2018-12-31', '2019-12-31',  7, dtype='datetime64[D]')
            idx = pd.DatetimeIndex(d)
            ```
        * Pandas dataframe with Datetime Index
        ```python
        data = np.random.randn(3,2)
        cols = ['A','B']
        index = pd.DatetimeIndex(np.arange('2018-12-31', '2019-12-31', 3, dtype='datetime64[D]'))

        df = pd.Dataframe(data, cols, index)

        df.index
        df.index.max()
        df.index.argmax()
        ```
## Resampling
### What is resampling?
To represent the current data with a different frequency. For example, current dataset is daily, update the data to weekly, or monthly, etc.
### Why resampling?
* Problem framing: if your data is not available at the same frequency that you want to make predictions.
* Feature Engineering: provide additional structure or insight into the learning problem for supervised learning models
### Two types of resampling
* Up-sampling: to higher frequency, could be missing data
* Down-sampling: to lower frequency
### Two functions of resampling - down-sampling
* resample()
    * Aggregates data based on specified frequency and aggregation function.
    * Aggregation: sum, mean, max, etc.
* asfreq()
    * Selects data based on the specified frequency and returns the value at the end of the specified interval.
### Two functions of resample - up-sampling
* fill missing data with forward fill (ffill)
* fill missing data with backward fill (bfil)

### Resampling project 
[code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/TimeSeries_Resampling.ipynb)
## Shifting
### What is Shifting?
A common operation on time-series data is to shift or "lag" the values back and forward in time, such as to calculate percentage change from sample to sample 
* Series.shift(self, periods=1, freq=None, axis=0, fill_value=None)
### Why shifting?
* Historic comparison
* Prediction
* Example: ROI 

### Two types of Shifting
* shift: shifts the data
* tshift: shifts the time index

### Shifting Project
[code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/TimeSeries_Shifting.ipynb)
## Rolling and Expanding
### What is Rolling?
Rolling-window analysis of a time-series model assesses:
* The stability of the model over time. A common time-series model assumption is that the coefficients are constant with respect to time. Checking for instability amounts to examining whether the coefficients are time-invariant.
* The forecast accuracy of the model.
divide the data into "windows" of time, and then calculate an aggregate function for each window. In this way we obtain a simple moving average. 

### Why rolling?
* Access stability
* Forecasting accuracy

### Methods of rolling
```python
Series.rolling(self, window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None)
```

### Combine grouping with rolling
```python
df.groupby('col1').rolling(2).sum() # cauclate sum of two consecutive elements
df.groupby('col1').expanding().sum() # Accumulative sum
```


### Expanding
Taking into account from the start point to the end point

### Rolling Expanding Project
[Code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/TimeSeries_RollingExpanding.ipynb)

## Visualization of time series data
* Plotting
* set plot axises, title, legend, autoscale, etc
* Set xlimits, ylimits
    * by arguments: df['col'].plot(figsize=(12,6), xlim=['2020-01-01','2020-01-31], ylim=[20,50])
    * by slicing data set: df['col']['2012-01-01':'2012-12-01' ]
* Color and style
* X Ticks
* Major vs. Minor Axis Values
* Gridlines

### Time Series Visualization and Formating Project 
[Code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/TimeSeries_VisualizationFormating.ipynb)

# Part 5: Time Series Analysis with Statsmodels
## Properties of Time Series Data:
* Trends: upward, downward, horizontal/stationary
* Seasonality: Repeating trends
* Cyclical: Trends with no set repetition
## Introduction: Statsmodels module 
* H-P filter seperates time series data <a href="https://www.codecogs.com/eqnedit.php?latex=y_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{t}" title="y_{t}" /></a> into a trend compoent <a href="https://www.codecogs.com/eqnedit.php?latex=\tau_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_{t}" title="\tau_{t}" /></a> and cyclical component <a href="https://www.codecogs.com/eqnedit.php?latex=c_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{t}" title="c_{t}" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=y_{t}&space;=&space;\tau&space;_{t}&space;&plus;&space;c_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{t}&space;=&space;\tau&space;_{t}&space;&plus;&space;c_{t}" title="y_{t} = \tau _{t} + c_{t}" /></a>

* Determined by minimizing quadratic loss function, where <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> is a smoothing parameter
<a href="https://www.codecogs.com/eqnedit.php?latex=min_{\tau&space;_{t}}\sum&space;_{t=1}^{T}{c_{t}}^{2}&plus;\lambda&space;\sum&space;_{t=1}^{T}[(\tau&space;_{t}&space;-&space;\tau&space;_{t-1})&space;-&space;(\tau&space;_{t-1}&space;-&space;\tau&space;_{t-2})]^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{\tau&space;_{t}}\sum&space;_{t=1}^{T}{c_{t}}^{2}&plus;\lambda&space;\sum&space;_{t=1}^{T}[(\tau&space;_{t}&space;-&space;\tau&space;_{t-1})&space;-&space;(\tau&space;_{t-1}&space;-&space;\tau&space;_{t-2})]^{2}" title="min_{\tau _{t}}\sum _{t=1}^{T}{c_{t}}^{2}+\lambda \sum _{t=1}^{T}[(\tau _{t} - \tau _{t-1}) - (\tau _{t-1} - \tau _{t-2})]^{2}" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> has default values:
    * 1600: quarterly data
    * 6.25: annual data
    * 129600: monthly daata
* Implement with statsmodels module with hpfilter
```python
from statsmodels.tsa.filters.hp_filter import hpfilter
gdp_cycle, gdp_trend = hpfilter(df['realgdp'], lamb=1600)
```
### Statsmodels HPFilter Project on Time Series
[Code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/Statsmodels_hpfilter.ipynb)

## ETS 
### What is ETS?
Error - Trend - Seasonality

### Models related to ETS
* Exponential Smothing
* Trend Methods Models
* ETS Decomposition

### ETS Decomposition
Statsmodels provides a seasonal decomposition tool to seperate different components
* eg. HPFilter: trend + cycle
* ETS models will take each term for smoothing and add operations. Then create a model to fit data
* Why ETS: visualize time series data with ETS is a good way to understand data behavior

### ETS for Time Series
* Additive model
When trend is linear and seasonality and trend seems to be constants
* Multiplicative model
Non linear rate

### Decomposition Project 
[Code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/Statsmodels_ETS.ipynb)

## EWMA Thoery
### SMA vs. EWMA
* SMA: Simple Moving Average
    * Entire model is constrained to the same window size
    * Disadvantage: 1. lag by size of window; 2. smaller window, larger noise; 3. never reach full peak or valley of data due to average; 4. does not predict future behavior, just describe trends; 5. Extreme historical values skew SMA significantly
    * .rolling()

* EWMA: Exponentially weighted moving average

    *Series.ewm(self, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0)
    * Recent data is weighted more than older data
        * Simple Exponential Smoothing:with one smoothing factor alpha; failed to other factors like trend and seasonality
        * Triple Exponential Smoothing
        * Holt-Winters Methods
### EWMA Simple Exponential Smoothing Project
[Code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/Statsmodel_EWMA.ipynb)

## Holt - Winters Methods Theory
### What is H_W method?
Holt Winter seasonal method: three smoothing equations:
* level: <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a>
* trend: <a href="https://www.codecogs.com/eqnedit.php?latex=\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /></a>
* seasonal component: <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a>

### Two variations:
Differ in the nature of the seasonal component
* Additive method
When seasonal variations are roughly constatnt through series
* multiplicative method
When seasonal variations are changing proportional to the level of the series

### Methods:
* Single Exponential Smoothing: smoothing factor alpha

![equation](https://latex.codecogs.com/gif.latex?y_{0}&space;=&space;x_{0}x&space;\rightarrow&space;y_{t}&space;=&space;(1-\alpha)y_{t-1}&space;&plus;&space;\alpha&space;x_{t})

* Double Exponential Smoothing: trend factor beta
    * level: ![equation](https://latex.codecogs.com/gif.latex?l_{t}&space;=&space;(1-\alpha)l_{t-1}&space;&plus;&space;\alpha&space;x_{t})
    * trend: ![equation](https://latex.codecogs.com/gif.latex?b_{t}&space;=&space;(1-\beta)b_{t-1}&space;&plus;&space;\beta&space;(l_{t}&space;-&space;l_{t-1}))
    * fitted model ![equeation](https://latex.codecogs.com/gif.latex?y_{t}&space;=&space;l_{t}&space;&plus;&space;b_{t})
    * forecasting model ![equation](https://latex.codecogs.com/gif.latex?\hat{y}_{t&plus;h}&space;=&space;l_{t}&space;&plus;&space;hb_{t})

* Triple Exponential Smoothing: gamma for seasonality
    * level: ![equation](https://latex.codecogs.com/gif.latex?l_{t}&space;=&space;(1-\alpha)l_{t-1}&space;&plus;&space;\alpha&space;x_{t})
    * trend: ![equation](https://latex.codecogs.com/gif.latex?b_{t}&space;=&space;(1-\beta)b_{t-1}&space;&plus;&space;\beta&space;(l_{t}&space;-&space;l_{t-1}))
    * seasonality: ![equation](https://latex.codecogs.com/gif.latex?c_{t}&space;=&space;(1-\gamma&space;)c_{t-L}&space;&plus;&space;\gamma(x_{t}&space;-&space;l_{t-1}&space;-&space;b_{t-1}))
    * fitted model: ![equation](https://latex.codecogs.com/gif.latex?y_{t}&space;=&space;(l_{t}&space;&plus;&space;b_{t})c_{t})
    * prediction: ![equation](https://latex.codecogs.com/gif.latex?\hat{y}_{t&plus;m}&space;=&space;(l_{t}&space;&plus;&space;mb_{t})c_{t-L&plus;1&plus;(m-1)modL})
        * L: number of divisions per cycle
### Double and Triple Exponential Moving Average Project
[Code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/Statsmodel_HoltWinters_EWMA.ipynb)

# Part 6: General Forecasting Models
## Tools for Time Series Data analysis
* Pandas, Numpy, Statsmodels
## Methods to Model Time Series Behavior
* HP filter: trend + cyclical
* ETS: error trend seasonality
* Simple Moving Average
* Simple Exponential Moving Average
* Double Exponential Moving Average
* Triple Exponential Moving Average
## Forecasting Time Series Data
Precedure: Choose Model > Split Data > Fit Model on Train set > Evaluate Model on Test set > Re-fit model on entire data set > forecast future data

## Introduction
### Build model and predict

### Evaluation:
* MAE: Average residuals, won't alert if forecasting points were way off
* MSE: Units
* RMSE: 


### Stationary dataset


### Intorduction to Forecasting model project
[code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/Forecasting_Intro.ipynb)

## ACF: AutoCorrelation function plot
### Correlation:
* (-1, 1)
* strength of linear relationship

### AutoCorrelation plot: Correlogram
* Shows correlation of the series with itself lagged by x time units
* y is correlation, x is number of time units lagged
* Why? to answer question like how correlated are today's sales to yesterday's sales
* Typical features:
    * gradual decline
    * sharp drop off

### PACF Partial AutoCorrelation function plot
Only describes the direct relationshiop between an obeservation and its lag

### Why important?
help choose order parameters for ARIMA based models

### ACF PACF Project
[Code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/Forcasting_ACF_PACF.ipynb)
## AR
* In moving average model in Holt-Winters, using linear combination of predictors (level, trend, seasonal)

### Autoregssion
* in Autoregression model, forecast using a linear combination of past values of the variable. Autoregression run against a set of lagged values of order p.
* the output variable depends linearly on its own previous values and on a stockastic term

### function
* AR(1)
![equation](https://latex.codecogs.com/gif.latex?y_{t}&space;=&space;c&space;&plus;&space;\phi&space;y_{t-1}&space;&plus;&space;\varepsilon&space;_{t})
* AR(2)
![equation](https://latex.codecogs.com/gif.latex?y_{t}&space;=&space;c&space;&plus;&space;\phi&space;_{1}y_{t-1}&plus;&space;\phi&space;_{2}y_{t-2}&space;&plus;&space;\varepsilon&space;_{t})
* Higher oreder AR models become complex. Might take into noise.
* AR(p)
Let the statsmodels to choose the order p


### AutoRegression with statsmodel.tsa.ar_model project
[Code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/Forecasting_AR.ipynb)

## Descriptive Statistics
To understand the attributes of time series data
### Stationary
    * Dickey-Fuller test
        * performs a test of classic null hypothesis test and return p value
        * p < 0.05 reject null ---> dataset is stationary

### Causality
    * Granger Causality Test
        * determin if one time series is useful in forecasting another

### Evalute forecasts:
    * MAE
    * MSE
    * RMSE
    * AIC
        * evalute a collection of models
        * Penalties are provided for the number of parameters used to prevent overfitting
    * BIC
        * AIC using a Bayesian approach
### Seaonality Plots
### Descriptive Statistics of Time Series Data and model project
[code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/Forecasting_Descriptive%20Statistics.ipynb)

## ARIMA
### Overview
* one of the most common time series models
* ARIMA is not capable of perfectly predicting any time series data. For example: stock price, depending on time and many other factors
* ARIMA works well when working with a time series where data is directly related to time stamp, such as airline passenger data
* Purpse:
    * Better understand data
    * Future prediction
* Types:
    * Non seasonal ARIMA
    * Seasonal ARIMA
* data: Non stationary dataset
    * intial differencing step -> integrated to eliminate non-stationary
* Compenents:
    * P: AR autoregression
        * A regression model that utilizes the depenent relationship between a current observation and obervations over a previous period
        * evolving variable of interest is regressed on its own lagged
    * d: I Integrated
        * Differencing of observations( obervation - previous obervation) to make data time series stationary
    * q: MA Moving Average
        * use dependency between an observation and a residual from a moving average model applied to laged observations
        * regression error is linear combination of error terms whose values occurred contemporaneously and at various time in past
        * smooth out noise from time series

* Stationary: constant mean and variance overtime
    * allow model to predice mean and variance, covaraiance, be the same in the future
    * Augmented Dickey-Fuller test
    * if data is non-stationary, need to transform it to stationary (integrate, differencing) to evaluate it
        * seasonal data: difference by season
        * seasonal ARIMA: taking seasonal difference of first difference
* How to choose p, d, q
    * Mehtod 1: ACP PACP
    * Method 2: Grid Search

### ARMA(No differencing term)

### Choosing ARIMA Orders: p,d,q
* If autocorrlection plot shows positive autocorrelation at first lag (lag-1), then it suggest to use AR terms in relation to lag
* if negative, then use MA terms
* p: # of lag observations 
* d: # times that raw observations are differenced
* q: size of moving average window (order of moving average)
### How to determine? ACF/PACF
* Typically a shorp drop after lag 'K' of PACF, suggested AR(k) model should be used
* Ifthere is a gradual decline， use MA model


### Greid search p, d, q combinations
* pmdarima: 
    * Use AIC as metric to compare models
### Project Choosing ARIMA Order
[code](https://github.com/xiaomiaoright/TimeSeriesProjects/blob/master/Forecasting_ChoosingARIMAOrders.ipynb)

## 
# Part 7: Deep Learning for Time Series Forecasting

# part *8: Facebook's Prophet Library
