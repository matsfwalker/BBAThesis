# Stability of Factor-Loadings in the Fama-French 5-Factors Model 

## ESADE BBA 2026
*Mats Walker, Frederik Tiefenbacher, Laurenz Köpp*

## Note that this project is work-in-progress and subject to change

This repository contains the code for the empirical analysis of the Bachelor's thesis *"Stability of Factor-Loadings in the Fama-French 5-Factors Model"*. The thesis examines the evolution of the factor-loadings (betas) of industry portfolios in the US from 2008-2026 with a special emphasis on the aftermath of the COVID pandemic.

## Research Question
To what extent are factor loadings (betas) in the Fama–French five-factor model stable over time, particularly in the post-COVID period, and what do observed changes imply for the model’s continued validity?

- RQ1: Do factor loadings exhibit statistically significant breaks around COVID-19? Are observed changes persistent or mean-reverting?

- RQ2: How do beta stability and factor exposures differ across sectors and firm size groups?

- RQ3: To what extent can post-COVID changes in factor loadings be explained by underlying economic and sectoral transformations, and what do they imply for the future applicability of the Fama–French five-factor model?

## Repository structure

The repository is split into 5 main directories:
- [analysis](analysis/): This directory contains the analysis of the findings and visualising them through charts
- [src](src/): Python modules in which the data processing and model building are implemented
- [configs](configs/): The directory where the configurations of the project are defined. These are the only definitions that may be changed when running the project
- [data](data/): The repository in which the data (raw, processed and portfolio data) is found as .csv files
- [results](results/): The repository in which the data of the results and plots can be found

## Data used
The data for the project is downloaded from WRDS (Wharton Research Data Services), specifically the compustat library. 

Due to licensing restrictions, raw WRDS data is not included in the project. To reproduce the datasets, run [src/download_data.py](src/download_data.py) and [src/data_cleaning.py](src/data_cleaning.py) 

Raw data is processed in the project to fill missing (primarily non-trading days) market information and remove non-significant or non-active firms.