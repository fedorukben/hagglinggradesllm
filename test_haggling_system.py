#!/usr/bin/env python3
"""
Test script for the Haggling Grading System
"""

from haggling_grading_system import HagglingGradingSystem

def test_haggling_system():
    """Test the haggling system with sample data"""
    
    print("Testing Haggling Grading System")
    print("=" * 50)
    
    # Initialize the system
    system = HagglingGradingSystem(max_rounds=5, convergence_threshold=3)
    
    # Test grade conversion
    print("Testing Grade Scale Conversion:")
    print("-" * 30)
    test_grades = [0, 5, 10, 15]
    for grade in test_grades:
        converted = system.convert_grade_scale(grade)
        print(f"  {grade} (0-15) → {converted} (0-100)")
    
    # Test data
    test_essay = """
    Technology has become an integral part of modern life. Computers and smartphones 
    are everywhere, from homes to schools to workplaces. While some people worry that 
    technology makes us lazy or isolated, I believe the benefits outweigh the drawbacks.
    
    First, technology makes work more efficient. Tasks that used to take hours can now 
    be completed in minutes. For example, writing a paper is much faster with word 
    processing software than with a typewriter. Communication is also easier - we can 
    send emails instantly instead of waiting for letters to arrive.
    
    Second, technology connects people. Social media allows us to stay in touch with 
    friends and family who live far away. Video calls make it possible to see and talk 
    to people across the world. This actually brings people together, not apart.
    
    Some people argue that technology makes us less social, but I disagree. Technology 
    is just a tool - how we use it is up to us. If we choose to use it to connect with 
    others, it can actually improve our relationships.
    
    In conclusion, technology has many benefits that make our lives better. It helps 
    us work more efficiently and stay connected with others. While there are some 
    concerns about overuse, the positive aspects far outweigh the negative ones.
    """
    
    test_prompt = "Discuss the impact of technology on modern society. Do the benefits outweigh the drawbacks?"
    
    print("\nTest Essay:")
    print("-" * 30)
    print(test_essay[:200] + "...")
    print()
    
    print("Test Prompt:")
    print("-" * 30)
    print(test_prompt)
    print()
    
    # Test the system
    print("Running haggling analysis...")
    actual_grade_0_15 = 12.0  # Assume actual grade of 12 on 0-15 scale
    result = system.process_essay(999, test_prompt, test_essay, actual_grade_0_15)
    
    print("\n" + "=" * 50)
    print("HAGGLING RESULTS")
    print("=" * 50)
    
    print(f"Essay ID: {result['id']}")
    print(f"Actual Grade: {result['actual_grade_0_15']} (0-15) → {result['actual_grade_0_100']} (0-100)")
    print(f"Baseline Grade: {result['baseline_grade']}")
    print(f"Initial Generous Grade: {result['initial_generous_grade']}")
    print(f"Initial Harsh Grade: {result['initial_harsh_grade']}")
    print(f"Final Generous Grade: {result['final_generous_grade']}")
    print(f"Final Harsh Grade: {result['final_harsh_grade']}")
    print(f"Consensus Grade: {result['consensus_grade']}")
    print(f"Rounds Used: {result['rounds_used']}")
    print(f"Converged: {result['converged']}")
    print(f"Final Grade Difference: {result['grade_difference']}")
    
    # Calculate accuracy metrics
    baseline_error = abs(result['baseline_grade'] - result['actual_grade_0_100'])
    generous_error = abs(result['initial_generous_grade'] - result['actual_grade_0_100'])
    harsh_error = abs(result['initial_harsh_grade'] - result['actual_grade_0_100'])
    consensus_error = abs(result['consensus_grade'] - result['actual_grade_0_100'])
    
    print(f"\nAccuracy Analysis:")
    print(f"  Baseline error: {baseline_error:.1f} points")
    print(f"  Generous error: {generous_error:.1f} points")
    print(f"  Harsh error: {harsh_error:.1f} points")
    print(f"  Consensus error: {consensus_error:.1f} points")
    
    best_approach = min([('baseline', baseline_error), ('generous', generous_error), ('harsh', harsh_error), ('consensus', consensus_error)], key=lambda x: x[1])
    print(f"  Best approach: {best_approach[0]} (error: {best_approach[1]:.1f})")
    
    # Show improvement
    if consensus_error < baseline_error:
        improvement = baseline_error - consensus_error
        print(f"  Haggling improves over baseline by {improvement:.1f} points")
    else:
        degradation = consensus_error - baseline_error
        print(f"  Haggling degrades from baseline by {degradation:.1f} points")
    
    print("\nHaggling History:")
    print("-" * 30)
    for round_data in result['haggling_history']:
        print(f"Round {round_data['round']}: Generous={round_data['generous']}, Harsh={round_data['harsh']}")
    
    print("\n✓ Haggling system test completed successfully!")

if __name__ == "__main__":
    test_haggling_system()
