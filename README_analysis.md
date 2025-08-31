# Feynman Grading System - Statistical Analysis

This directory contains a comprehensive statistical analysis framework for evaluating the performance of multiple LLMs in the Feynman grading system.

## Overview

The analysis evaluates the performance of 5 LLMs (ChatGPT, Claude, Gemini, Grok, and Qwen) across 250 essays, measuring:
- Grade improvements from baseline to final grades
- Readability success rates (Flesch-Kincaid and Dale-Chall)
- Iteration efficiency
- Performance across different essay quality levels
- Statistical significance and practical impact

## Files

- `statistical_analysis.py` - Main analysis script
- `requirements_analysis.txt` - Python dependencies
- `README_analysis.md` - This file

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_analysis.txt
```

## Usage

### Running the Complete Analysis

To run the full statistical analysis pipeline:

```bash
python statistical_analysis.py
```

This will execute all analysis components and generate:
- Comprehensive console output with statistical results
- Visualizations saved as `feynman_analysis_results.png`
- Summary files: `llm_performance_summary.csv` and `correlation_matrix.csv`

### Analysis Components

The script performs the following analyses:

1. **Descriptive Statistics & Data Quality Assessment**
   - Dataset overview and missing data analysis
   - Grade distributions and category breakdowns
   - Iteration and readability summary statistics

2. **Comparative Analysis Across LLMs**
   - Performance summary by LLM
   - One-way ANOVA and post-hoc tests
   - Effect size calculations (eta-squared)
   - Non-parametric alternatives (Kruskal-Wallis)

3. **Readability Analysis**
   - Readability score distributions
   - Success rate analysis by LLM
   - Chi-square tests for readability success
   - Correlation analysis

4. **Essay Quality Stratification Analysis**
   - Performance across low/medium/high quality essays
   - Two-way ANOVA for LLM × quality interactions
   - Category-specific performance metrics

5. **Advanced Statistical Modeling**
   - Linear regression models for grade improvement prediction
   - Iteration count prediction models
   - Feature importance analysis

6. **Reliability & Consistency Analysis**
   - Inter-LLM correlation analysis
   - Grade consistency assessment
   - Intra-class correlation coefficients

7. **Practical Significance Assessment**
   - Effect size calculations (Cohen's d)
   - Educational impact thresholds
   - Cost-benefit analysis

8. **Visualizations**
   - Box plots for grade improvements
   - Scatter plots for baseline vs final grades
   - Success rate bar charts
   - Iteration distribution histograms
   - Performance heat maps

## Output Files

### Console Output
The script provides detailed statistical results including:
- Descriptive statistics
- Statistical test results (ANOVA, chi-square, etc.)
- Effect sizes and confidence intervals
- Practical significance assessments

### Generated Files
- `feynman_analysis_results.png` - Comprehensive visualization dashboard
- `llm_performance_summary.csv` - Detailed LLM performance metrics
- `correlation_matrix.csv` - Correlation analysis results

## Key Metrics Analyzed

### Grade Improvement
- Baseline to final grade changes
- Statistical significance across LLMs
- Effect sizes and practical impact

### Readability Success
- Flesch-Kincaid score ≤ 2.4 (target: young children)
- Dale-Chall score ≤ 12.9 (target: young children)
- Overall success rates by LLM

### Efficiency Metrics
- Iteration count analysis
- Processing efficiency
- Optimal iteration strategies

### Quality Stratification
- Performance across essay quality levels
- LLM specialization patterns
- Quality-specific recommendations

## Statistical Rigor

The analysis employs:
- **Parametric tests**: One-way ANOVA, two-way ANOVA, linear regression
- **Non-parametric alternatives**: Kruskal-Wallis, Mann-Whitney U
- **Post-hoc analysis**: Tukey's HSD, Bonferroni corrections
- **Effect size measures**: Cohen's d, eta-squared, R²
- **Reliability measures**: Intra-class correlation, inter-rater agreement

## Educational Impact Assessment

The analysis evaluates practical significance using:
- Grade improvement thresholds (small: 0.5, medium: 1.0, large: 2.0 points)
- Readability success rates for young learners
- Cost-benefit analysis of iteration strategies
- Recommendations for educational implementation

## Customization

You can modify the analysis by:
- Adjusting readability thresholds in the `__init__` method
- Adding new statistical tests in the respective methods
- Modifying visualization layouts in `create_visualizations()`
- Extending the analysis with additional metrics

## Troubleshooting

### Common Issues
1. **Missing dependencies**: Install all packages from `requirements_analysis.txt`
2. **Data file not found**: Ensure `feynman_grading_results_250.csv` is in the same directory
3. **Memory issues**: For large datasets, consider processing in chunks
4. **Visualization errors**: Ensure matplotlib backend is properly configured

### Data Requirements
The analysis expects a CSV file with columns:
- `id`: Essay identifier
- `llm`: LLM name (CHATGPT, CLAUDE, GEMINI, GROK, QWEN)
- `baseline_grade`: Initial grade
- `final_grade`: Final grade after iterations
- `actual_grade`: Human-assigned grade
- `final_flesch_kincaid`: Final Flesch-Kincaid score
- `final_dale_chall`: Final Dale-Chall score
- `iterations_used`: Number of iterations
- `core_concept`: Extracted core concept
- `final_explanation`: Final simplified explanation

## Citation

If you use this analysis framework in your research, please cite:
```
Feynman Grading System Statistical Analysis Framework
Statistical Analysis Team, 2024
```

## Support

For questions or issues with the analysis, please refer to the main project documentation or create an issue in the repository.
