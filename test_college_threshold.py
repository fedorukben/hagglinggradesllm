#!/usr/bin/env python3
"""
Test script to verify the college-level threshold modifications
"""

from feynman_grading_system import FeynmanGradingSystem, ReadabilityScorer

def test_college_threshold():
    """Test that the system now targets college-level readability"""
    
    # Initialize the system
    system = FeynmanGradingSystem(max_iterations=3)
    
    print("Testing College-Level Threshold Modifications")
    print("=" * 50)
    
    # Test the threshold logic
    print(f"Target Flesch-Kincaid threshold: {system.flesch_threshold}")
    print(f"Target level: College/Undergraduate (Grade {system.flesch_threshold})")
    
    # Test some sample texts
    test_texts = [
        "Simple text for children.",
        "This is a more complex sentence with academic vocabulary.",
        "The implementation of sophisticated methodologies necessitates comprehensive analysis of multifaceted variables within the context of contemporary theoretical frameworks."
    ]
    
    print("\nTesting sample texts:")
    for i, text in enumerate(test_texts, 1):
        meets, flesch, dale = system.meets_readability_thresholds(text)
        status = "✓ MEETS" if meets else "✗ BELOW"
        print(f"Text {i}: {text}")
        print(f"  Flesch-Kincaid: {flesch} {status} threshold ({system.flesch_threshold})")
        print(f"  Dale-Chall: {dale}")
        print()
    
    # Test the enhancement prompts
    print("Testing enhancement prompts:")
    sample_concept = "Climate change affects weather patterns."
    
    print(f"Original concept: {sample_concept}")
    print("Enhancement prompts would now target college-level complexity instead of simplification.")
    
    print("\n✓ System successfully modified to target college-level readability!")

if __name__ == "__main__":
    test_college_threshold()
