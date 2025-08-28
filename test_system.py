from feynman_grading_system import FeynmanGradingSystem, ReadabilityScorer

def test_readability_scorer():
    """Test the readability scoring functionality"""
    print("Testing Readability Scorer...")
    
    scorer = ReadabilityScorer()
    
    # Test text
    test_text = "This is a simple test sentence. It has basic words and structure."
    
    flesch = scorer.flesch_kincaid(test_text)
    dale_chall = scorer.dale_chall(test_text)
    
    print(f"Test text: {test_text}")
    print(f"Flesch-Kincaid: {flesch}")
    print(f"Dale-Chall: {dale_chall}")
    
    # Test thresholds
    system = FeynmanGradingSystem()
    meets_threshold, f_score, d_score = system.meets_readability_thresholds(test_text)
    print(f"Meets target age level: {meets_threshold}")
    print(f"Target: Flesch-Kincaid ≤ {system.flesch_threshold} (grade {system.flesch_threshold})")
    
    print("Readability test complete!\n")

def test_single_essay():
    """Test processing a single essay"""
    print("Testing Single Essay Processing...")
    
    # Sample essay data
    essay_id = 1
    prompt = "University: Some people believe that university students should be required to attend classes. Others believe that going to classes should be optional for students. Which point of view do you agree with? Use specific reasons and details to explain your answer."
    essay = "Many university classes requires the attendance in the class so that it is included in the score. However, I think that the class attendance should be optional for students. There are three reasons to argue optional attendance. First, the adults have freedom to make own decisions. The university students are matured enough to be responsible to their decisions. Less attendance will results into less opportunity to learn. If a student decided to take the risk, the successive results is all his or her matter. Second, it is not efficient to have the attendance as one of the scoring criteria. The score is a metric to measure how the student understand the course materials. If a student already has perfect understanding, he or she should have good score. There is no need to attend the class only for grading. Therefore, nice grading system should not include the attendance. Third, the professor can instruct the students who are more willing to learn. Not all students want to learn a lot in the class. They just attend the class, and do something else. However, it makes the class atmosphere worse. Free attendance can solve the problem by releasing just-attenders. It will leave the students who want to learn something in the class. In short, the attendance is not really important in the university. The students are aged enough to choose own attendance. Moreover, the attendance causes several inefficiencies. The class can be more effective without the attendance."
    actual_grade = 11.0
    
    system = FeynmanGradingSystem(max_iterations=3)  # Limit iterations for testing
    
    print(f"Processing essay {essay_id}...")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Essay length: {len(essay)} characters")
    print(f"Actual grade: {actual_grade}")
    
    # Test with ChatGPT only for now
    from code import LLM
    
    try:
        result = system.process_essay(essay_id, prompt, essay, actual_grade, LLM.CHATGPT)
        print("\nResult:")
        for key, value in result.items():
            if key in ['core_concept', 'final_explanation']:
                print(f"  {key}: {str(value)[:100]}...")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error processing essay: {e}")
    
    print("Single essay test complete!\n")

if __name__ == "__main__":
    print("="*60)
    print("FEYNMAN GRADING SYSTEM - TEST MODE")
    print("="*60)
    
    test_readability_scorer()
    test_single_essay()
    
    print("All tests complete!")
