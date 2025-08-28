from feynman_grading_system import FeynmanGradingSystem

def main():
    """Run the full Feynman grading system on a small sample"""
    print("Starting Full Feynman Grading System - Age-Based Approach")
    print("="*60)
    
    # Initialize system with age-based thresholds
    system = FeynmanGradingSystem(max_iterations=6)
    
    # Run analysis on first 5 essays
    input_file = "DREsS_New.tsv"
    output_file = "feynman_grading_results_sample.csv"
    max_essays = 5  # Start with 5 essays for testing
    
    print(f"Processing first {max_essays} essays with all 3 LLMs...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Max iterations: {system.max_iterations}")
    print(f"Target age level: Flesch-Kincaid ≤ {system.flesch_threshold} (grade {system.flesch_threshold})")
    print(f"Approach: Iteratively lower target age until readability threshold is met")
    
    system.run_complete_analysis(input_file, output_file, max_essays)

if __name__ == "__main__":
    main()
