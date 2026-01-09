# HydroSystem
This package aims to bundle a variety of utilities related to rainfall-runoff modelling.

# Design Principles
The fundamental unit of analysis for hydrological modelling are individual time series. The canonical representation for
these is in the form of pandas.Series or pandas.DataFrame. This package does not attempt to reinvent these tools,
but rather serves as a convenient set of utilities for manipulating these.

# Features
Below is given an overview of the utilities available in the package.

### Discharge Modelling
The package includes a family of discharge models which operate on precipitation, temperature and evapotranspiration
in order to produce discharge estimates.

### Extreme Value Analysis
The package includes a family of statistical models for extremes that can be fitted to individual time series.

### Trend Analysis
The package includes an interface to the Mann-kendal test and Sen's robust slope estimator.

### Statistical Tests
The package includes an interface that enables t-testing between different timeseries.

### Plotting
The package provides a bunch of convenient plotting facilities based on matplotlib.

### Data
The package comes with some sample data, which is used to demonstrate the functionalities available.
