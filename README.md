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
* Create a date/datetime/time object

"""python
from datetime import datetime
my_date_time = datetime(my_year,my_month,my_day,my_hour,my_minute,my_second)
my_date_time.day
my_date_time.hour
"""
* Create datetime range
* create DateTime Index
## Resampling

## Shifting

## Rolling and Expanding

## Visualization

### Related Code: [here]

# Part 5: Time Series Analysis with Statsmodels

# Part 6: General Forecasting Models

# Part 7: Deep Learning for Time Series Forecasting

# part *8: Facebook's Prophet Library
