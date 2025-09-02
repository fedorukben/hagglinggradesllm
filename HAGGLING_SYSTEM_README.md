# Haggling Grading System

## Overview

The Haggling Grading System implements a novel approach to automated essay grading where two AI models with different grading philosophies negotiate back and forth until they reach a consensus grade. This system mimics the process of having multiple human graders discuss and compromise on a final grade.

## How It Works

### The Two Models

1. **Lenient Model (ChatGPT)**: A generous grader who:
   - Focuses on what students did well
   - Is encouraging and supportive
   - Considers effort and improvement potential
   - Gives the benefit of the doubt
   - Uses the full 0-100 scale but leans toward higher scores

2. **Strict Model (Claude)**: A critical grader who:
   - Maintains high academic standards
   - Is critical of weaknesses
   - Expects excellence and precision
   - Focuses on what's missing or incorrect
   - Uses the full 0-100 scale but is conservative

### The Haggling Process

1. **Initial Grades**: Both models independently grade the essay on a 0-100 scale
2. **Negotiation Rounds**: The models take turns adjusting their grades based on:
   - The other model's current grade
   - Previous rounds of negotiation
   - Their own grading philosophy
   - The quality of the essay
3. **Convergence**: The process continues until:
   - The grades are within a specified threshold (default: 3 points)
   - Maximum rounds are reached (default: 8 rounds)
4. **Final Grade**: The consensus grade is calculated as the average of the final grades

## Key Features

### Grade Scale
- **0-100 integer scale**: More granular than the original 0-15 scale
- **Convergence threshold**: Configurable tolerance for agreement
- **Maximum rounds**: Prevents infinite negotiation

### Haggling Strategies
- **Stand your ground**: Models maintain their philosophy when justified
- **Small concessions**: Models make incremental adjustments
- **Historical context**: Previous rounds inform current decisions
- **Philosophy preservation**: Models stay true to their core approach

### Output Metrics
- Initial grades from both models
- Final grades after negotiation
- Consensus grade
- Number of rounds used
- Whether convergence was achieved
- Final grade difference between models

## Usage

### Basic Usage

```python
from haggling_grading_system import HagglingGradingSystem

# Initialize the system
system = HagglingGradingSystem(max_rounds=8, convergence_threshold=3)

# Process a single essay
result = system.process_essay(essay_id, prompt, essay_text, actual_grade)

# Run full analysis
system.run_analysis("data.tsv", "results.csv")
```

### Configuration Options

```python
system = HagglingGradingSystem(
    max_rounds=10,           # Maximum negotiation rounds
    convergence_threshold=2   # Grade difference for convergence
)
```

### Running the System

1. **Test the system**:
   ```bash
   python test_haggling_system.py
   ```

2. **Run on sample data**:
   ```bash
   python haggling_grading_system.py
   ```

3. **Run on custom data**:
   ```python
   system = HagglingGradingSystem()
   system.run_analysis("your_data.tsv", "your_results.csv", max_essays=100)
   ```

## Expected Output

### Console Output
```
Processing essay 123 with haggling system...
  Getting initial grades...
    Initial lenient grade: 85
    Initial strict grade: 62
  Round 1...
    Lenient: 85 → 82
    Strict: 62 → 65
  Round 2...
    Lenient: 82 → 80
    Strict: 65 → 68
  Round 3...
    Lenient: 80 → 79
    Strict: 68 → 70
    Grades converged! Lenient: 79, Strict: 70
  Final consensus grade: 75
```

### Results File
The system generates a CSV file with columns:
- `id`: Essay identifier
- `initial_lenient_grade`: Lenient model's first grade
- `initial_strict_grade`: Strict model's first grade
- `final_lenient_grade`: Lenient model's final grade
- `final_strict_grade`: Strict model's final grade
- `consensus_grade`: Final agreed-upon grade
- `actual_grade`: Human-assigned grade
- `rounds_used`: Number of negotiation rounds
- `converged`: Whether models reached agreement
- `grade_difference`: Final difference between models

## Advantages

1. **Multiple Perspectives**: Combines different grading philosophies
2. **Negotiation Process**: Mimics human grading discussions
3. **Consensus Building**: Forces models to justify and compromise
4. **Transparency**: Shows the negotiation process
5. **Flexibility**: Configurable parameters for different use cases

## Research Questions

This system can help answer:
- Do negotiated grades improve accuracy over single-model grades?
- How do different grading philosophies affect final consensus?
- What patterns emerge in the negotiation process?
- Does the haggling process reduce bias or introduce new biases?

## Files

- `haggling_grading_system.py`: Main system implementation
- `test_haggling_system.py`: Test script with sample data
- `HAGGLING_SYSTEM_README.md`: This documentation
- `haggling_grading_results_*.csv`: Output results files

## Future Enhancements

- **Multiple models**: Include more than two models
- **Weighted consensus**: Weight models differently based on accuracy
- **Negotiation strategies**: Implement different haggling algorithms
- **Real-time visualization**: Show negotiation progress graphically
- **Model personality**: Create more diverse grading personalities
