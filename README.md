# VolleyVision AI -- Smart Volleyball Player Performance Analysis System

A Data Science capstone project that analyses professional volleyball
attack statistics and converts raw match counts into meaningful
player-performance measures.

## Project Overview

VolleyVision AI is proposed as a smart player-performance analysis
system for volleyball. The complete concept could cover attacking,
serving, blocking, defence and reception. This capstone implements and
evaluates the **attack-analysis module** using men's Volleyball Nations
League (VNL) 2024 player statistics.

The project investigates:

-   Which factors are associated with an attacker's success rate?
-   Do players with more attacks achieve a higher success rate?
-   Do attack-success rates differ significantly across teams?
-   Can player statistics be used to create player profiles and a
    transparent performance index?

## Dataset

-   **Dataset:** VNL 2024 -- Men's Volleyball Stats
-   **Source:** Kaggle, compiled by `jonathanpmoyer`
-   **File used:** `VNL2024Men_Attackers.csv`
-   **Original data:** 294 players across 16 teams, with 8 columns.
-   **Main analysis:** 161 players with at least 30 recorded attacks.

The report identifies 62 players with zero attacks and excludes them
because attack-rate measures cannot be calculated for them. A further 71
players with 1--29 attacks are excluded from the main analysis because
rates based on very few attempts can be unstable.

> Before publishing or redistributing the dataset, confirm the Kaggle
> dataset's licence and terms of use. Add the exact dataset URL and
> access date here.

## Performance Measures

The analysis derives these measures from the attack counts:

  -----------------------------------------------------------------------
  Measure                             Calculation
  ----------------------------------- -----------------------------------
  Attack success (`kill_pct`)         Points ÷ total attacks × 100

  Attack error (`error_pct`)          Errors ÷ total attacks × 100

  Hitting efficiency (`efficiency`)   (Points − errors) ÷ total attacks ×
                                      100
  -----------------------------------------------------------------------

The source file also includes player, team, attack-count and per-match
statistics.

## Methodology

The project follows a five-phase data-science workflow:

1.  **Define the problem** -- establish the research question,
    objectives and unit of analysis.
2.  **Understand the data** -- load the CSV, inspect its structure and
    verify the attack totals.
3.  **Prepare and explore** -- handle zero-attack and low-volume
    records, derive performance measures, and explore distributions and
    relationships.
4.  **Analyse** -- apply statistical tests, regression and clustering.
5.  **Communicate** -- present findings, limitations and future scope.

### Analysis techniques

-   Python with pandas and NumPy for data loading, cleaning and
    calculations
-   Matplotlib and Seaborn for visualisation
-   Spearman rank correlation
-   Welch's t-test and Mann--Whitney U test
-   Kruskal--Wallis test for comparison across teams
-   Multiple linear regression with a 75/25 train-test split
    (`random_state=42`)
-   K-Means clustering (`k=3`) using standardised performance features
-   A custom VolleyVision Performance Index

### VolleyVision Performance Index

The report defines the index as:

``` text
Index = 50 + 10 × (
    0.5 × z(hitting efficiency)
  + 0.3 × z(points per match)
  + 0.2 × z(log(total attacks))
)
```

The weights are design choices intended to combine performance quality
and workload; they are not universal measures of player quality.

## Key Findings

For the 161 players with at least 30 attacks:

-   Mean attack success rate: **50.2%**
-   Mean attack error rate: **14.0%**
-   Mean hitting efficiency: **36.2%**
-   Error rate had a strong negative association with attack success
    (`Spearman ρ = -0.61`, `p < 0.001`).
-   Attack volume had a weak negative association with success
    (`ρ = -0.16`, `p = 0.048`).
-   No statistically significant difference in attack success was found
    across the 16 teams (`Kruskal–Wallis p = 0.171`).
-   The regression model explained approximately **35%** of the
    variation in success rate on the held-out test set (`R² = 0.35`).
-   K-Means produced three descriptive player profiles; the silhouette
    score was 0.38.

These are associations in this dataset and should not be interpreted as
proof of cause and effect.

## Tools and Technologies

-   Python 3
-   pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   SciPy
-   scikit-learn
-   Google Colab / Jupyter Notebook

## How to Run

1.  Download or clone this repository.

2.  Open the project notebook (`.ipynb`) in Google Colab or Jupyter
    Notebook.

3.  Upload `VNL2024Men_Attackers.csv` when prompted, or place it in the
    notebook's expected data directory.

4.  Install any missing dependencies:

    ``` bash
    pip install pandas numpy matplotlib seaborn scipy scikit-learn
    ```

5.  Run the notebook cells in order, from data loading through analysis
    and visualisation.

**Note:** The uploaded project report describes the analysis but does
not include the notebook or CSV in this README package. Update the
notebook and dataset filenames above if your repository uses different
names.

## Suggested Repository Structure

``` text
VolleyVision-AI/
├── README.md
├── capstone_project.ipynb
├── data/
│   └── VNL2024Men_Attackers.csv
└── report/
    └── Project_Report_VolleyVision_AI.docx
```

Only include the dataset in the repository if its licence permits
redistribution. Otherwise, provide instructions for obtaining it from
its original Kaggle page.

## Limitations

-   The analysis uses attack statistics only; serving, blocking,
    defence, reception and setting are outside its scope.
-   The dataset does not include player position, height or age.
-   The data covers one tournament, so results may differ in other
    seasons or competitions.
-   The 30-attack qualification threshold is a judgement call, although
    the report checks alternative thresholds of 20 and 50 attacks.
-   Success and error percentages share the same denominator, which
    partly contributes mechanically to their relationship.
-   The regression model does not include factors such as set quality,
    opposing blocks or player roles.
-   The clusters are moderately separated and are descriptive rather
    than definitive player categories.

## Future Scope

-   Add serving, blocking, defence, reception, setting and other
    available player-statistics tables.
-   Build an interactive dashboard where coaches can inspect player
    measures, profiles and index scores.
-   Analyse match-by-match performance and consistency.
-   Extend the analysis to women's competitions and other seasons.
-   Explore predictive models when more relevant variables become
    available.

## References

-   `jonathanpmoyer` (2024). *VNL 2024 -- Men's Volleyball Stats*
    [Dataset](#dataset). Kaggle. **Add the exact Kaggle URL, access date
    and licence before final publication.**
-   Virtanen, P. et al. (2020). "SciPy 1.0: Fundamental Algorithms for
    Scientific Computing in Python." *Nature Methods*, 17, 261--272.
-   Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
    Python." *Journal of Machine Learning Research*, 12, 2825--2830.
-   McKinney, W. (2010). "Data Structures for Statistical Computing in
    Python." *Proceedings of the 9th Python in Science Conference*.
