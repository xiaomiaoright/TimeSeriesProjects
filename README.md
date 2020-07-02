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

### Related Code: [here]

# Part 5: Time Series Analysis with Statsmodels

# Part 6: General Forecasting Models

# Part 7: Deep Learning for Time Series Forecasting

# part *8: Facebook's Prophet Library
