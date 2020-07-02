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

## Shifting

## Rolling and Expanding

## Visualization

### Related Code: [here]

# Part 5: Time Series Analysis with Statsmodels

# Part 6: General Forecasting Models

# Part 7: Deep Learning for Time Series Forecasting

# part *8: Facebook's Prophet Library
