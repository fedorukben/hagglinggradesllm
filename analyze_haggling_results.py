#!/usr/bin/env python3
"""
Analysis script for Haggling Grading System results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def analyze_haggling_results(results_file: str):
    """Analyze haggling grading results"""
    
    print("Haggling Grading System Analysis")
    print("=" * 50)
    
    # Load results
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} results from {results_file}")
    
    # Basic statistics
    print("\nBASIC STATISTICS")
    print("-" * 30)
    print(f"Essays processed: {len(df)}")
    print(f"Average baseline grade: {df['baseline_grade'].mean():.1f}")
    print(f"Average actual grade (0-15): {df['actual_grade_0_15'].mean():.2f}")
    print(f"Average actual grade (0-100): {df['actual_grade_0_100'].mean():.2f}")
    print(f"Average consensus grade: {df['consensus_grade'].mean():.2f}")
    print(f"Average rounds used: {df['rounds_used'].mean():.2f}")
    print(f"Convergence rate: {df['converged'].mean()*100:.1f}%")
    
    # Accuracy analysis
    print("\nACCURACY ANALYSIS")
    print("-" * 30)
    
    # Calculate errors
    baseline_error = abs(df['baseline_grade'] - df['actual_grade_0_100'])
    generous_error = abs(df['initial_generous_grade'] - df['actual_grade_0_100'])
    harsh_error = abs(df['initial_harsh_grade'] - df['actual_grade_0_100'])
    consensus_error = abs(df['consensus_grade'] - df['actual_grade_0_100'])
    
    print(f"Baseline (simple heuristic):")
    print(f"  Average error: {baseline_error.mean():.2f} points")
    print(f"  RMSE: {np.sqrt((baseline_error**2).mean()):.2f}")
    print(f"  Bias: {df['baseline_grade'].mean() - df['actual_grade_0_100'].mean():.2f}")
    
    print(f"\nGenerous model:")
    print(f"  Average error: {generous_error.mean():.2f} points")
    print(f"  RMSE: {np.sqrt((generous_error**2).mean()):.2f}")
    print(f"  Bias: {df['initial_generous_grade'].mean() - df['actual_grade_0_100'].mean():.2f}")
    
    print(f"\nHarsh model:")
    print(f"  Average error: {harsh_error.mean():.2f} points")
    print(f"  RMSE: {np.sqrt((harsh_error**2).mean()):.2f}")
    print(f"  Bias: {df['initial_harsh_grade'].mean() - df['actual_grade_0_100'].mean():.2f}")
    
    print(f"\nConsensus (haggling):")
    print(f"  Average error: {consensus_error.mean():.2f} points")
    print(f"  RMSE: {np.sqrt((consensus_error**2).mean()):.2f}")
    print(f"  Bias: {df['consensus_grade'].mean() - df['actual_grade_0_100'].mean():.2f}")
    
    # Determine best approach
    approaches = [
        ('Baseline', baseline_error.mean(), np.sqrt((baseline_error**2).mean())),
        ('Generous', generous_error.mean(), np.sqrt((generous_error**2).mean())),
        ('Harsh', harsh_error.mean(), np.sqrt((harsh_error**2).mean())),
        ('Consensus', consensus_error.mean(), np.sqrt((consensus_error**2).mean()))
    ]
    
    best_mae = min(approaches, key=lambda x: x[1])
    best_rmse = min(approaches, key=lambda x: x[2])
    
    print(f"\nBest approach by MAE: {best_mae[0]} ({best_mae[1]:.2f})")
    print(f"Best approach by RMSE: {best_rmse[0]} ({best_rmse[2]:.2f})")
    
    # Improvement analysis
    print("\nIMPROVEMENT ANALYSIS")
    print("-" * 30)
    
    # Calculate improvement from baseline to consensus
    baseline_improvement = baseline_error - consensus_error
    generous_improvement = generous_error - consensus_error
    harsh_improvement = harsh_error - consensus_error
    
    print(f"Baseline → Consensus improvement: {baseline_improvement.mean():.2f} points")
    print(f"Generous model improvement: {generous_improvement.mean():.2f} points")
    print(f"Harsh model improvement: {harsh_improvement.mean():.2f} points")
    print(f"Overall improvement: {(generous_improvement + harsh_improvement).mean()/2:.2f} points")
    
    # Show if haggling is worth it
    if consensus_error.mean() < baseline_error.mean():
        improvement_pct = (baseline_error.mean() - consensus_error.mean()) / baseline_error.mean() * 100
        print(f"Haggling improves over baseline by {improvement_pct:.1f}%")
    else:
        degradation_pct = (consensus_error.mean() - baseline_error.mean()) / baseline_error.mean() * 100
        print(f"Haggling degrades from baseline by {degradation_pct:.1f}%")
    
    # Haggling effectiveness analysis
    print("\nHAGGLING EFFECTIVENESS")
    print("-" * 30)
    
    # Calculate improvement from initial to consensus
    generous_improvement = generous_error - consensus_error
    harsh_improvement = harsh_error - consensus_error
    
    print(f"Generous model improvement: {generous_improvement.mean():.2f} points")
    print(f"Harsh model improvement: {harsh_improvement.mean():.2f} points")
    print(f"Overall improvement: {(generous_improvement + harsh_improvement).mean()/2:.2f} points")
    
    # Grade distribution analysis
    print("\nGRADE DISTRIBUTION ANALYSIS")
    print("-" * 30)
    
    print(f"Actual grades range: {df['actual_grade_0_100'].min():.1f} - {df['actual_grade_0_100'].max():.1f}")
    print(f"Baseline grades range: {df['baseline_grade'].min():.1f} - {df['baseline_grade'].max():.1f}")
    print(f"Generous grades range: {df['initial_generous_grade'].min():.1f} - {df['initial_generous_grade'].max():.1f}")
    print(f"Harsh grades range: {df['initial_harsh_grade'].min():.1f} - {df['initial_harsh_grade'].max():.1f}")
    print(f"Consensus grades range: {df['consensus_grade'].min():.1f} - {df['consensus_grade'].max():.1f}")
    
    # Convergence analysis
    print("\nCONVERGENCE ANALYSIS")
    print("-" * 30)
    
    converged = df[df['converged'] == True]
    not_converged = df[df['converged'] == False]
    
    print(f"Converged essays: {len(converged)} ({len(converged)/len(df)*100:.1f}%)")
    print(f"Not converged essays: {len(not_converged)} ({len(not_converged)/len(df)*100:.1f}%)")
    
    if len(converged) > 0:
        print(f"Average rounds for converged: {converged['rounds_used'].mean():.2f}")
    if len(not_converged) > 0:
        print(f"Average rounds for not converged: {not_converged['rounds_used'].mean():.2f}")
    
    # Quality of convergence
    print(f"Average final grade difference: {df['grade_difference'].mean():.2f}")
    print(f"Final grade difference range: {df['grade_difference'].min():.1f} - {df['grade_difference'].max():.1f}")
    
    # Performance by grade level
    print("\nPERFORMANCE BY GRADE LEVEL")
    print("-" * 30)
    
    # Split into low, medium, high grades
    df['grade_level'] = pd.cut(df['actual_grade_0_100'], 
                               bins=[0, 33, 67, 100], 
                               labels=['Low (0-33)', 'Medium (34-67)', 'High (68-100)'])
    
    for level in df['grade_level'].unique():
        if pd.isna(level):
            continue
        level_data = df[df['grade_level'] == level]
        level_baseline_error = abs(level_data['baseline_grade'] - level_data['actual_grade_0_100'])
        level_consensus_error = abs(level_data['consensus_grade'] - level_data['actual_grade_0_100'])
        
        print(f"{level}:")
        print(f"  Essays: {len(level_data)}")
        print(f"  Baseline error: {level_baseline_error.mean():.2f}")
        print(f"  Consensus error: {level_consensus_error.mean():.2f}")
        print(f"  Improvement: {level_baseline_error.mean() - level_consensus_error.mean():.2f}")
    
    # Create summary table
    print("\nSUMMARY TABLE")
    print("-" * 30)
    summary_data = {
        'Metric': ['MAE', 'RMSE', 'Bias', 'Range'],
        'Baseline': [
            f"{baseline_error.mean():.2f}",
            f"{np.sqrt((baseline_error**2).mean()):.2f}",
            f"{df['baseline_grade'].mean() - df['actual_grade_0_100'].mean():.2f}",
            f"{df['baseline_grade'].max() - df['baseline_grade'].min():.1f}"
        ],
        'Generous': [
            f"{generous_error.mean():.2f}",
            f"{np.sqrt((generous_error**2).mean()):.2f}",
            f"{df['initial_generous_grade'].mean() - df['actual_grade_0_100'].mean():.2f}",
            f"{df['initial_generous_grade'].max() - df['initial_generous_grade'].min():.1f}"
        ],
        'Harsh': [
            f"{harsh_error.mean():.2f}",
            f"{np.sqrt((harsh_error**2).mean()):.2f}",
            f"{df['initial_harsh_grade'].mean() - df['actual_grade_0_100'].mean():.2f}",
            f"{df['initial_harsh_grade'].max() - df['initial_harsh_grade'].min():.1f}"
        ],
        'Consensus': [
            f"{consensus_error.mean():.2f}",
            f"{np.sqrt((consensus_error**2).mean()):.2f}",
            f"{df['consensus_grade'].mean() - df['actual_grade_0_100'].mean():.2f}",
            f"{df['consensus_grade'].max() - df['consensus_grade'].min():.1f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Final verdict
    print(f"\nFINAL VERDICT")
    print("-" * 30)
    
    if consensus_error.mean() < baseline_error.mean():
        improvement = baseline_error.mean() - consensus_error.mean()
        print(f"✓ Haggling system IMPROVES over baseline by {improvement:.2f} points")
        print(f"✓ Worth the computational cost and complexity")
    else:
        degradation = consensus_error.mean() - baseline_error.mean()
        print(f"✗ Haggling system DEGRADES from baseline by {degradation:.2f} points")
        print(f"✗ Simple baseline may be more effective")
    
    if consensus_error.mean() < min(generous_error.mean(), harsh_error.mean()):
        print(f"✓ Consensus improves over individual models")
    else:
        print(f"✗ Consensus does not improve over individual models")

def main():
    """Main function"""
    # Look for haggling results files
    import glob
    results_files = glob.glob("haggling_grading_results_*.csv")
    
    if not results_files:
        print("No haggling results files found. Please run the haggling system first.")
        return
    
    # Use the most recent file
    latest_file = max(results_files, key=lambda x: os.path.getmtime(x))
    print(f"Analyzing results from: {latest_file}")
    
    analyze_haggling_results(latest_file)

if __name__ == "__main__":
    import os
    main()
