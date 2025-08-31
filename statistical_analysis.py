"""
Comprehensive Statistical Analysis for Feynman Grading System Results
====================================================================

This script implements a robust statistical analysis plan to evaluate the performance
of multiple LLMs in the Feynman grading system across 250 essays.

Author: Statistical Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd, kruskal, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FeynmanStatisticalAnalysis:
    """
    Comprehensive statistical analysis for Feynman grading system results.
    """
    
    def __init__(self, data_path='feynman_grading_results_250.csv'):
        """Initialize the analysis with data loading and preprocessing."""
        self.data_path = data_path
        self.df = None
        self.llms = ['CHATGPT', 'CLAUDE', 'GEMINI', 'GROK', 'QWEN']
        self.readability_thresholds = {
            'final_flesch_kincaid': 2.4,
            'final_dale_chall': 12.9
        }
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """Load the grading results data."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records")
        
    def preprocess_data(self):
        """Preprocess the data for analysis."""
        print("Preprocessing data...")
        
        # Handle missing values
        self.df['essay_missing'] = self.df['core_concept'].str.contains('nan', na=False)
        
        # Calculate grade improvement
        self.df['grade_improvement'] = self.df['final_grade'] - self.df['baseline_grade']
        
        # Create grade categories
        self.df['grade_category'] = pd.cut(
            self.df['actual_grade'], 
            bins=[0, 5, 10, 15], 
            labels=['Low (0-5)', 'Medium (6-10)', 'High (11-15)'],
            include_lowest=True
        )
        
        # Calculate readability success
        self.df['flesch_success'] = self.df['final_flesch_kincaid'] <= self.readability_thresholds['final_flesch_kincaid']
        self.df['dale_chall_success'] = self.df['final_dale_chall'] <= self.readability_thresholds['final_dale_chall']
        self.df['readability_success'] = self.df['flesch_success'] & self.df['dale_chall_success']
        
        print("Data preprocessing complete!")
        
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics."""
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS & DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Dataset overview
        print(f"\nDataset Overview:")
        print(f"Total records: {len(self.df)}")
        print(f"Unique essays: {self.df['id'].nunique()}")
        print(f"LLMs tested: {len(self.llms)}")
        print(f"Missing essays: {self.df['essay_missing'].sum()}")
        
        # Grade distributions
        print(f"\nGrade Distributions:")
        print(f"Actual grades - Mean: {self.df['actual_grade'].mean():.2f}, Std: {self.df['actual_grade'].std():.2f}")
        print(f"Baseline grades - Mean: {self.df['baseline_grade'].mean():.2f}, Std: {self.df['baseline_grade'].std():.2f}")
        print(f"Final grades - Mean: {self.df['final_grade'].mean():.2f}, Std: {self.df['final_grade'].std():.2f}")
        
        # Grade category distribution
        print(f"\nGrade Category Distribution:")
        category_dist = self.df['grade_category'].value_counts().sort_index()
        for category, count in category_dist.items():
            print(f"  {category}: {count} essays ({count/len(self.df)*100:.1f}%)")
            
        # Iteration analysis
        print(f"\nIteration Analysis:")
        print(f"Mean iterations: {self.df['iterations_used'].mean():.2f}")
        print(f"Max iterations: {self.df['iterations_used'].max()}")
        print(f"Min iterations: {self.df['iterations_used'].min()}")
        
        # Readability analysis
        print(f"\nReadability Analysis:")
        print(f"Flesch-Kincaid success rate: {self.df['flesch_success'].mean()*100:.1f}%")
        print(f"Dale-Chall success rate: {self.df['dale_chall_success'].mean()*100:.1f}%")
        print(f"Overall readability success: {self.df['readability_success'].mean()*100:.1f}%")
        
    def llm_comparative_analysis(self):
        """Perform comparative analysis across LLMs."""
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS ACROSS LLMs")
        print("="*60)
        
        # Summary statistics by LLM
        llm_stats = self.df.groupby('llm').agg({
            'baseline_grade': ['mean', 'std'],
            'final_grade': ['mean', 'std'],
            'grade_improvement': ['mean', 'std'],
            'iterations_used': ['mean', 'std'],
            'readability_success': 'mean',
            'final_flesch_kincaid': 'mean',
            'final_dale_chall': 'mean'
        }).round(3)
        
        print(f"\nLLM Performance Summary:")
        print(llm_stats)
        
        # Statistical tests for grade improvement
        print(f"\nStatistical Tests for Grade Improvement:")
        
        # Prepare data for ANOVA
        llm_groups = [self.df[self.df['llm'] == llm]['grade_improvement'].values for llm in self.llms]
        
        # ANOVA test
        f_stat, p_value = f_oneway(*llm_groups)
        print(f"One-way ANOVA:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Effect size (eta-squared)
        ss_between = sum(len(group) * (group.mean() - self.df['grade_improvement'].mean())**2 for group in llm_groups)
        ss_total = sum((val - self.df['grade_improvement'].mean())**2 for val in self.df['grade_improvement'])
        eta_squared = ss_between / ss_total
        print(f"  Effect size (eta-squared): {eta_squared:.4f}")
        
        # Post-hoc analysis
        if p_value < 0.05:
            print(f"\nPost-hoc Analysis (Tukey's HSD):")
            tukey_result = tukey_hsd(*llm_groups)
            print(tukey_result)
            
        # Non-parametric alternative (Kruskal-Wallis)
        kw_stat, kw_p = kruskal(*llm_groups)
        print(f"\nKruskal-Wallis Test (non-parametric):")
        print(f"  H-statistic: {kw_stat:.4f}")
        print(f"  p-value: {kw_p:.6f}")
        
    def readability_analysis(self):
        """Analyze readability scores and success rates."""
        print("\n" + "="*60)
        print("READABILITY ANALYSIS")
        print("="*60)
        
        # Readability distributions
        print(f"\nReadability Score Distributions:")
        print(f"Flesch-Kincaid - Mean: {self.df['final_flesch_kincaid'].mean():.2f}, Std: {self.df['final_flesch_kincaid'].std():.2f}")
        print(f"Dale-Chall - Mean: {self.df['final_dale_chall'].mean():.2f}, Std: {self.df['final_dale_chall'].std():.2f}")
        
        # Success rates by LLM
        readability_by_llm = self.df.groupby('llm')['readability_success'].agg(['mean', 'count']).round(3)
        print(f"\nReadability Success Rates by LLM:")
        print(readability_by_llm)
        
        # Chi-square test for readability success
        contingency_table = pd.crosstab(self.df['llm'], self.df['readability_success'])
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-square test for readability success:")
        print(f"  Chi-square statistic: {chi2:.4f}")
        print(f"  p-value: {chi2_p:.6f}")
        print(f"  Degrees of freedom: {dof}")
        
        # Correlation analysis
        correlations = self.df[['grade_improvement', 'final_flesch_kincaid', 'final_dale_chall', 'iterations_used']].corr()
        print(f"\nCorrelation Matrix:")
        print(correlations.round(3))
        
    def stratification_analysis(self):
        """Analyze performance across different essay quality levels."""
        print("\n" + "="*60)
        print("ESSAY QUALITY STRATIFICATION ANALYSIS")
        print("="*60)
        
        # Performance by grade category
        category_performance = self.df.groupby(['llm', 'grade_category']).agg({
            'grade_improvement': ['mean', 'std'],
            'readability_success': 'mean',
            'iterations_used': 'mean'
        }).round(3)
        
        print(f"\nPerformance by Essay Quality Level:")
        print(category_performance)
        
        # Two-way ANOVA for grade improvement
        from scipy.stats import f_oneway
        from statsmodels.stats.anova import anova_lm
        from statsmodels.formula.api import ols
        
        # Prepare data for two-way ANOVA
        model_data = self.df[['llm', 'grade_category', 'grade_improvement']].dropna()
        model = ols('grade_improvement ~ C(llm) + C(grade_category) + C(llm):C(grade_category)', data=model_data).fit()
        anova_table = anova_lm(model, typ=2)
        
        print(f"\nTwo-way ANOVA Results:")
        print(anova_table.round(4))
        
    def advanced_modeling(self):
        """Perform advanced statistical modeling."""
        print("\n" + "="*60)
        print("ADVANCED STATISTICAL MODELING")
        print("="*60)
        
        # Prepare data for modeling
        model_df = self.df.dropna(subset=['grade_improvement', 'baseline_grade', 'iterations_used'])
        
        # Feature engineering
        model_df['llm_encoded'] = pd.Categorical(model_df['llm']).codes
        model_df['grade_category_encoded'] = pd.Categorical(model_df['grade_category']).codes
        
        # Model 1: Predict grade improvement
        print(f"\nModel 1: Predicting Grade Improvement")
        X1 = model_df[['llm_encoded', 'baseline_grade', 'iterations_used', 'grade_category_encoded']]
        y1 = model_df['grade_improvement']
        
        model1 = LinearRegression()
        model1.fit(X1, y1)
        
        y1_pred = model1.predict(X1)
        r2_1 = r2_score(y1, y1_pred)
        rmse_1 = np.sqrt(mean_squared_error(y1, y1_pred))
        
        print(f"  R² Score: {r2_1:.4f}")
        print(f"  RMSE: {rmse_1:.4f}")
        print(f"  Coefficients: {dict(zip(X1.columns, model1.coef_))}")
        
        # Model 2: Predict iteration count
        print(f"\nModel 2: Predicting Iteration Count")
        X2 = model_df[['llm_encoded', 'baseline_grade', 'grade_category_encoded']]
        y2 = model_df['iterations_used']
        
        model2 = LinearRegression()
        model2.fit(X2, y2)
        
        y2_pred = model2.predict(X2)
        r2_2 = r2_score(y2, y2_pred)
        rmse_2 = np.sqrt(mean_squared_error(y2, y2_pred))
        
        print(f"  R² Score: {r2_2:.4f}")
        print(f"  RMSE: {rmse_2:.4f}")
        print(f"  Coefficients: {dict(zip(X2.columns, model2.coef_))}")
        
    def reliability_analysis(self):
        """Analyze reliability and consistency across LLMs."""
        print("\n" + "="*60)
        print("RELIABILITY & CONSISTENCY ANALYSIS")
        print("="*60)
        
        # Intra-class correlation for final grades
        from scipy.stats import pearsonr
        
        # Calculate correlations between LLMs for final grades
        # Handle duplicate entries by taking the first occurrence
        llm_final_grades = self.df.drop_duplicates(subset=['id', 'llm']).pivot(index='id', columns='llm', values='final_grade')
        
        print(f"\nInter-LLM Correlations (Final Grades):")
        correlation_matrix = llm_final_grades.corr()
        print(correlation_matrix.round(3))
        
        # Average correlation
        corr_values = []
        available_llms = [llm for llm in self.llms if llm in llm_final_grades.columns]
        for i in range(len(available_llms)):
            for j in range(i+1, len(available_llms)):
                llm1, llm2 = available_llms[i], available_llms[j]
                # Get common non-null values
                common_data = llm_final_grades[[llm1, llm2]].dropna()
                if len(common_data) > 1:
                    corr, _ = pearsonr(common_data[llm1], common_data[llm2])
                    corr_values.append(corr)
        
        if corr_values:
            print(f"Average inter-LLM correlation: {np.mean(corr_values):.4f}")
        else:
            print("Average inter-LLM correlation: Cannot calculate (insufficient data)")
        
        # Consistency analysis
        print(f"\nGrade Consistency Analysis:")
        for llm in self.llms:
            llm_data = self.df[self.df['llm'] == llm]
            if len(llm_data) > 1:
                baseline_final_corr, _ = pearsonr(llm_data['baseline_grade'], llm_data['final_grade'])
                print(f"  {llm}: Baseline-Final correlation = {baseline_final_corr:.4f}")
            else:
                print(f"  {llm}: Insufficient data for correlation")
            
    def practical_significance(self):
        """Assess practical significance of results."""
        print("\n" + "="*60)
        print("PRACTICAL SIGNIFICANCE ASSESSMENT")
        print("="*60)
        
        # Effect size calculations
        print(f"\nEffect Size Analysis:")
        
        # Cohen's d for grade improvements
        overall_mean_improvement = self.df['grade_improvement'].mean()
        overall_std_improvement = self.df['grade_improvement'].std()
        
        for llm in self.llms:
            llm_improvement = self.df[self.df['llm'] == llm]['grade_improvement']
            cohens_d = (llm_improvement.mean() - overall_mean_improvement) / overall_std_improvement
            print(f"  {llm} Cohen's d: {cohens_d:.4f}")
            
        # Educational impact assessment
        print(f"\nEducational Impact Assessment:")
        
        # Grade improvement thresholds
        small_effect = 0.5  # Half a grade point
        medium_effect = 1.0  # One grade point
        large_effect = 2.0   # Two grade points
        
        llms_above_threshold = {}
        for threshold, effect_name in [(small_effect, 'small'), (medium_effect, 'medium'), (large_effect, 'large')]:
            llms_above_threshold[effect_name] = []
            for llm in self.llms:
                llm_improvement = self.df[self.df['llm'] == llm]['grade_improvement'].mean()
                if llm_improvement >= threshold:
                    llms_above_threshold[effect_name].append(llm)
            print(f"  LLMs with {effect_name} effect (≥{threshold}): {llms_above_threshold[effect_name]}")
            
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feynman Grading System - Statistical Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Grade improvement by LLM
        ax1 = axes[0, 0]
        sns.boxplot(data=self.df, x='llm', y='grade_improvement', ax=ax1)
        ax1.set_title('Grade Improvement by LLM')
        ax1.set_xlabel('LLM')
        ax1.set_ylabel('Grade Improvement')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Baseline vs Final grades
        ax2 = axes[0, 1]
        for llm in self.llms:
            llm_data = self.df[self.df['llm'] == llm]
            ax2.scatter(llm_data['baseline_grade'], llm_data['final_grade'], 
                       alpha=0.6, label=llm, s=20)
        ax2.plot([0, 15], [0, 15], 'k--', alpha=0.5, label='No Change')
        ax2.set_title('Baseline vs Final Grades')
        ax2.set_xlabel('Baseline Grade')
        ax2.set_ylabel('Final Grade')
        ax2.legend()
        
        # 3. Readability success rates
        ax3 = axes[0, 2]
        readability_success = self.df.groupby('llm')['readability_success'].mean()
        readability_success.plot(kind='bar', ax=ax3, color='skyblue')
        ax3.set_title('Readability Success Rate by LLM')
        ax3.set_xlabel('LLM')
        ax3.set_ylabel('Success Rate')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Iteration count distribution
        ax4 = axes[1, 0]
        sns.histplot(data=self.df, x='iterations_used', hue='llm', multiple="stack", ax=ax4)
        ax4.set_title('Iteration Count Distribution')
        ax4.set_xlabel('Iterations Used')
        ax4.set_ylabel('Count')
        
        # 5. Performance by grade category
        ax5 = axes[1, 1]
        category_perf = self.df.groupby(['llm', 'grade_category'])['grade_improvement'].mean().unstack()
        category_perf.plot(kind='bar', ax=ax5)
        ax5.set_title('Grade Improvement by Quality Level')
        ax5.set_xlabel('LLM')
        ax5.set_ylabel('Mean Grade Improvement')
        ax5.tick_params(axis='x', rotation=45)
        ax5.legend(title='Grade Category')
        
        # 6. Readability scores
        ax6 = axes[1, 2]
        readability_data = self.df[['llm', 'final_flesch_kincaid', 'final_dale_chall']].melt(
            id_vars=['llm'], var_name='Metric', value_name='Score')
        sns.boxplot(data=readability_data, x='llm', y='Score', hue='Metric', ax=ax6)
        ax6.set_title('Readability Scores by LLM')
        ax6.set_xlabel('LLM')
        ax6.set_ylabel('Score')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('feynman_analysis_results.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'feynman_analysis_results.png'")
        
    def generate_report(self):
        """Generate a comprehensive statistical report."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        # Create summary statistics
        summary_stats = {
            'total_essays': self.df['id'].nunique(),
            'total_evaluations': len(self.df),
            'llms_tested': len(self.llms),
            'missing_essays': self.df['essay_missing'].sum(),
            'mean_grade_improvement': self.df['grade_improvement'].mean(),
            'std_grade_improvement': self.df['grade_improvement'].std(),
            'mean_iterations': self.df['iterations_used'].mean(),
            'readability_success_rate': self.df['readability_success'].mean()
        }
        
        # Best performing LLM
        llm_performance = self.df.groupby('llm')['grade_improvement'].mean().sort_values(ascending=False)
        best_llm = llm_performance.index[0]
        best_improvement = llm_performance.iloc[0]
        
        # Most consistent LLM
        llm_consistency = self.df.groupby('llm')['grade_improvement'].std().sort_values()
        most_consistent_llm = llm_consistency.index[0]
        
        # Best readability LLM
        llm_readability = self.df.groupby('llm')['readability_success'].mean().sort_values(ascending=False)
        best_readability_llm = llm_readability.index[0]
        
        print(f"\nEXECUTIVE SUMMARY:")
        print(f"  Total essays analyzed: {summary_stats['total_essays']}")
        print(f"  Total evaluations: {summary_stats['total_evaluations']}")
        print(f"  LLMs tested: {summary_stats['llms_tested']}")
        print(f"  Overall mean grade improvement: {summary_stats['mean_grade_improvement']:.3f}")
        print(f"  Overall readability success rate: {summary_stats['readability_success_rate']:.1%}")
        
        print(f"\nKEY FINDINGS:")
        print(f"  Best performing LLM: {best_llm} (improvement: {best_improvement:.3f})")
        print(f"  Most consistent LLM: {most_consistent_llm}")
        print(f"  Best readability LLM: {best_readability_llm}")
        
        # Save detailed results
        self.save_detailed_results()
        
    def save_detailed_results(self):
        """Save detailed analysis results to files."""
        
        # Save summary statistics
        summary_df = self.df.groupby('llm').agg({
            'grade_improvement': ['mean', 'std', 'min', 'max'],
            'readability_success': 'mean',
            'iterations_used': ['mean', 'std'],
            'final_flesch_kincaid': 'mean',
            'final_dale_chall': 'mean'
        }).round(3)
        
        summary_df.to_csv('llm_performance_summary.csv')
        
        # Save correlation matrix
        numeric_cols = ['baseline_grade', 'final_grade', 'grade_improvement', 
                       'final_flesch_kincaid', 'final_dale_chall', 'iterations_used']
        correlation_matrix = self.df[numeric_cols].corr()
        correlation_matrix.to_csv('correlation_matrix.csv')
        
        print("Detailed results saved to:")
        print("  - llm_performance_summary.csv")
        print("  - correlation_matrix.csv")
        
    def run_complete_analysis(self):
        """Run the complete statistical analysis pipeline."""
        print("FEYNMAN GRADING SYSTEM - COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)
        
        # Run all analysis components
        self.descriptive_statistics()
        self.llm_comparative_analysis()
        self.readability_analysis()
        self.stratification_analysis()
        self.advanced_modeling()
        self.reliability_analysis()
        self.practical_significance()
        self.create_visualizations()
        self.generate_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)

def main():
    """Main function to run the analysis."""
    # Initialize and run analysis
    analyzer = FeynmanStatisticalAnalysis()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
