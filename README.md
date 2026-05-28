# Stability of Factor-Loadings in the Fama-French 5-Factors Model

## ESADE BBA 2026
*Mats Walker, Frederik Tiefenbacher, Laurenz Köpp*

## Note that this project is work-in-progress and subject to change

This repository contains the code for the empirical analysis of the Bachelor's thesis *"Stability of Factor-Loadings in the Fama-French 5-Factors Model"*. The thesis examines the evolution of the factor-loadings (betas) of industry portfolios in the US from 2008-2026 with a special emphasis on the aftermath of the COVID pandemic.

## Research Question
To what extent are factor loadings (betas) in the Fama-French five-factor model stable over time, particularly in the post-COVID period, and what do observed changes imply for the model's continued validity?

- RQ1: Do factor loadings exhibit statistically significant breaks around COVID-19? Are observed changes persistent or mean-reverting?

- RQ2: How do beta stability and factor exposures differ across sectors and firm size groups?

- RQ3: To what extent can post-COVID changes in factor loadings be explained by underlying economic and sectoral transformations, and what do they imply for the future applicability of the Fama-French five-factor model?

## Repository Structure

The repository is split into 5 main directories:
- [analysis](analysis/): This directory contains the analysis of the findings and visualising them through charts
- [src](src/): Python modules in which the data processing and model building are implemented
- [configs](configs/): The directory where the configurations of the project are defined. These are the only definitions that may be changed when running the project
- [data](data/): The repository in which the data (raw, processed and portfolio data) is found as .csv files
- [results](results/): The repository in which the data of the results and plots can be found
- [sql](sql/): The sql queries to download the data from WRDS
- [logging](logging/): Logs of the project to increase traceability (currently written to the console)

## Methodology

The empirical analysis covers US equity market data from June 2009 to February 2026, divided into four sub-periods defined by distinctive macroeconomic conditions: Post-Financial Crisis Recovery (June 2009 to January 2015), Pre-COVID Expansion (January 2015 to February 2020), COVID-19 Crisis (February 2020 to January 2023), and Post-COVID Recovery (January 2023 to January 2026). These periods were selected to reflect materially different monetary policy regimes, interest rate environments, and market volatility conditions.

Stock-level time-series data are sourced from the Compustat dataset via WRDS, covering 39,818 US-listed companies across five exchanges. Firms are classified into the Fama-French 48 industry portfolios using four-digit SIC codes. Factor return data are obtained from the Kenneth French data library.

Raw data is subjected to several filtering steps. Firms present for less than 50% of the total time span are removed to limit listing and delisting noise, at the cost of introducing a degree of survivorship bias. Non-US-listed equities, OTC-traded stocks, and those listed on non-regulated exchanges are excluded. A minimum market capitalisation of $10 million (inflation-adjusted) and a minimum share price of $1 are enforced to remove nano-cap and penny stocks. Monthly returns are clipped to +/-100% to limit the influence of single-period outliers. At the portfolio level, a minimum of 10 constituent firms per monthly entry and a minimum of 26 monthly observations per portfolio over the full period are required, yielding a final sample of 35 industry portfolios.

Portfolios are constructed using a one-period lagged, market-capitalisation-weighted aggregation, consistent with standard index construction practice. Ordinary Least Squares (OLS) regression is then applied to each portfolio's excess returns against the five Fama-French factors. Gauss-Markov assumptions are assessed and found to be partially violated: heteroscedasticity is present in 54% of portfolios under the Levene test, and non-normality of residuals is detected by the Jarque-Bera test. Both are expected for financial time-series data. Newey-West heteroscedasticity and autocorrelation-consistent (HAC) standard errors with a lag of 3 are used for all inference. A significance threshold of p = 0.05 (t-statistic of 2.0) is applied throughout.

Statistical inference on factor loading stability employs t-tests for period-to-period beta changes, cross-period significance counts, R-squared analysis, and standard error comparisons. Beta changes are further analysed through distributional measures including cross-sectional standard deviations and within-portfolio temporal standard deviations to characterise the nature of instability across factors.

## Data Used
The data for the project is downloaded from WRDS (Wharton Research Data Services), specifically the Compustat library.

Due to licensing restrictions, raw WRDS data is not included in the project. To reproduce the datasets, run [src/download_data.py](src/download_data.py) and [src/data_cleaning.py](src/data_cleaning.py).

Raw data is processed in the project to fill missing (primarily non-trading days) market information and remove non-significant or non-active firms.

In order to create industry portfolios, the SIC aggregation from Fama-French is used. Different numbers of portfolios can be chosen in the [configs](configs/).

## Key Findings

The analysis finds that factor loadings in the Fama-French Five-Factor Model are not time-invariant, and that instability has intensified in the post-COVID period rather than reverting to pre-pandemic levels.

The market risk factor (Mkt-RF) behaves closest to time-invariance. Its variation is consistent with sampling error, and any break observed during the pandemic did not persist into the recovery period. The size factor (SMB) displays stable aggregate behaviour, with instability concentrated in a small cluster of outlier industries rather than distributed across the cross-section.

The remaining three factors show more pronounced and persistent breaks. The profitability factor (RMW) experienced a statistically significant break around COVID-19, after which its cross-sectional relevance declined substantially, with fewer industries loading significantly on the factor in the post-COVID period. The value factor (HML) shows the opposite trajectory: its significance count has risen since the pandemic, with beta changes of increasing magnitude suggesting that the factor is absorbing return variation the rest of the model can no longer explain. The investment factor (CMA) exhibits the greatest overall instability, with extreme portfolio-specific structural breaks and a collapse in cross-sectional significance after the pandemic.

At the sector level, commodity-driven industries, particularly Precious Metals, drive the most extreme outlier behaviour in both CMA and HML, as their capital expenditure and book-to-market dynamics follow commodity price cycles rather than the broader cross-sectional factors. The Automobiles and Trucks sector shows structural breaks in both SMB and HML consistent with the disruption introduced by the electric vehicle transition.

The post-COVID deterioration in model fit, together with the elevated standard errors of factor loadings and the divergence between R-squared and adjusted R-squared in the most recent period, suggests that the Fama-French Five-Factor Model's explanatory power has structurally weakened in the current macroeconomic environment. Whether this reflects a temporary regime shift or a more permanent reconfiguration of factor-return relationships remains an open question for future research.

## AI
Please note that AI (ChatGPT, Copilot, and Claude) has been used in this project. Docstrings in the code were generated with AI assistance. This README was written by Claude (Anthropic).
