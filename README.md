# Business Forecasting with Python

## Overview
In this project, I examined 16 annoynmized properties for a major hotel chain. The goal was to identify how to best forecast the 4-weeks ahead occupancy rate for each property based on:   
  1. Historical demand;
  2. attributes (property type, etc);
  3. Available on-the-book data; and
  4. Any exogenous features I can extract.

 For my analysis, I examined five categories of forecasting models:  
   1. **Simple baselines**, which included the Naive and SeasonalNaive; 
   1. **Error trend seasonality**, where I examined the use of `AutoETS`;
   1. **Statistical models**, `AutoARIMA`;
   1. **Machine Learning models**, `LightGBM`;
   1. **Deep Learning models**, `AutoNHITS` and `AutoNBEATS`  ; 
   1. **Transformer models**, `TimeGPT`.

## Python Notebook
The Python Notebook containing my code, results, and insights is available at: [Python Hotel Forecasting Demo](https://colab.research.google.com/drive/1HKo7VRzrUcmY0nbYEUL9FPqgOe9dcTdu?usp=sharing).

## Business Insights

This project compares a bunch of modern time series forecasting models to figure out which one actually works best in a fast-changing business setting. The main goal was to test different approaches—Deep Learning, traditional statistical models, and classic Machine Learning—to see which one gives the most accurate predictions for things like future demand, inventory needs, or other key metrics.

After running cross-validation and evaluating everything using metrics like MAE and RMSE, here’s what stood out:

Deep Learning Models Perform the Best

Finding: Models like TimeGPT, NHITS, and NBEATS consistently had the lowest errors. They handled complicated patterns in the data—like multiple seasonalities and sudden shifts—much better than the other methods.
Why It Matters: Using the top model (TimeGPT) in production would likely lead to better demand planning, fewer stockouts, less excess inventory, and overall smoother operations.

Statistical Models Are a Solid Backup Option

Finding: Models such as AutoETS and AutoARIMA were surprisingly strong. They’re interpretable and reliable, and in some cases, they weren’t far behind Deep Learning models—especially when the data had clear patterns.
Use Case: These models are perfect for situations where you need low latency or want a dependable fallback model if the main system goes down.

Machine Learning Models Need Better Features

Finding: The LightGBM model didn’t perform as well, mostly because it only used basic date features. Without richer inputs, it couldn’t keep up with the deep learning or statistical models.
What This Means: To make tree-based ML models competitive, the business needs to invest in better feature engineering—things like adding promotional schedules, economic indicators, competitor pricing, and holiday/event flags. ML models depend heavily on good features, so domain knowledge becomes essential.

Key Packages Used: pandas, numpy, matplotlib / seaborn / plotly, statsmodels, neuralforecast / lightgbm

## Future Work & Next Steps

To move this from a prototype to something that can run in production, here are the recommended next steps:

Error Analysis & Explainability: Look closely at the time series where all models struggled. This helps identify outliers, missing variables, or unexpected events that weren’t captured.

Uncertainty Estimates: Add prediction intervals (like 80% and 95%) so the business can understand the risk around each forecast—not just the point prediction.

I wish I could have also implemented the cross validation more fluidly throughout the model allowing me to better analyze each hotel uniquely to figure out what the best model would be for each hotel. 






