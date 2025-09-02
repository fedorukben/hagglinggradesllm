#!/usr/bin/env python3
"""
Haggling Grading System

Two models with different grading philosophies negotiate back and forth
until they reach a mutually agreeable grade for an essay.
"""

import pandas as pd
import re
import csv
import os
from typing import Dict, List, Tuple, Optional
import time
from code import ask, LLM

class HagglingGradingSystem:
    """System where generous and harsh models haggle to reach consensus"""
    
    def __init__(self, max_rounds: int = 10, convergence_threshold: int = 2):
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold  # Grades must be within this range to converge
        
        # Define all available models
        self.models = [LLM.CHATGPT, LLM.CLAUDE, LLM.QWEN]
        
        # Default model assignment (can be overridden)
        self.generous_model = LLM.CHATGPT  # More generous grader
        self.harsh_model = LLM.CLAUDE    # More critical grader
    
    def convert_grade_scale(self, grade_0_15: float) -> float:
        """Convert grade from 0-15 scale to 0-100 scale"""
        # Simple linear conversion: multiply by 100/15 = 6.67
        return round(grade_0_15 * (100/15), 1)
    
    def convert_grade_scale_reverse(self, grade_0_100: float) -> float:
        """Convert grade from 0-100 scale to 0-15 scale"""
        # Simple linear conversion: divide by 100/15 = 6.67
        return round(grade_0_100 / (100/15), 1)
    
    def get_baseline_grade(self, essay: str, prompt: str) -> int:
        """Get a simple baseline grade - just a middle-of-the-road guess"""
        # Simple heuristic: count words and give a basic score
        word_count = len(essay.split())
        
        # Very simple baseline: more words = higher score, but cap it
        if word_count < 100:
            baseline = 40  # Poor
        elif word_count < 200:
            baseline = 60  # Fair
        elif word_count < 300:
            baseline = 75  # Good
        else:
            baseline = 85  # Very good
        
        # Add some randomness to make it more realistic
        import random
        baseline += random.randint(-10, 10)
        baseline = max(0, min(100, baseline))
        
        return baseline
        
    def get_initial_grade(self, essay: str, prompt: str, model: LLM, is_generous: bool) -> int:
        """Get initial grade from either generous or harsh model"""
        
        if is_generous:
            grading_prompt = f"""
You are a GENEROUS essay grader who sees the best in student work and rewards effort. 
You focus on what students accomplish and their potential for growth.

Please grade the following essay based on the given prompt.

PROMPT: {prompt}

ESSAY: {essay}

As a GENEROUS grader, you should:
- Focus on what the student accomplished, even if imperfect
- Reward effort and attempt, not just perfection
- Consider the student's perspective and challenges
- Give credit for partial understanding
- Use the full 0-100 scale, but lean toward higher scores
- Recognize that learning is a process
- Appreciate creativity and original thinking
- Be encouraging and supportive

Please provide a single integer grade from 0-100, where:
- 0-20: Extremely poor (only if completely off-topic or blank)
- 21-40: Poor but shows some effort
- 41-60: Fair effort, some good points
- 61-80: Good work with room for improvement
- 81-95: Very good work
- 96-100: Excellent work (give this generously)

Remember: Be GENEROUS and encouraging. If the student made any effort at all, 
start thinking in the 60-80 range. Only go below 40 if the essay is truly problematic.

Provide ONLY the numerical grade, no explanation.
"""
        else:
            grading_prompt = f"""
You are a HARSH essay grader who maintains rigorous academic standards. 
You expect excellence and are not easily satisfied.

Please grade the following essay based on the given prompt.

PROMPT: {prompt}

ESSAY: {essay}

As a HARSH grader, you should:
- Hold high academic standards
- Be critical of weaknesses and gaps
- Expect excellence and precision
- Focus on what's missing or incorrect
- Use the full 0-100 scale, but be conservative
- Demand clear arguments and evidence
- Penalize poor organization and grammar
- Require depth of analysis

Please provide a single integer grade from 0-100, where:
- 0-20: Unacceptable work
- 21-40: Poor quality, significant issues
- 41-60: Mediocre work, needs improvement
- 61-75: Acceptable but not impressive
- 76-90: Good work
- 91-100: Exceptional work only

Provide ONLY the numerical grade, no explanation.
"""
        
        try:
            response = ask(grading_prompt, model)
            # Extract numerical grade from response
            grade_match = re.search(r'\b(\d+)\b', response)
            if grade_match:
                grade = int(grade_match.group(1))
                # Ensure grade is within 0-100 range
                return max(0, min(100, grade))
            else:
                print(f"Warning: Could not extract grade from {model.name} response: {response}")
                return 70 if is_generous else 30  # Default fallback grades
        except Exception as e:
            print(f"Error getting initial grade from {model.name}: {e}")
            return 70 if is_generous else 30  # Default fallback grades
    
    def haggle_round(self, essay: str, prompt: str, generous_grade: int, harsh_grade: int, 
                    round_num: int, previous_rounds: List[Dict]) -> Tuple[int, int]:
        """One round of haggling between the two models"""
        
        # Create context of previous rounds
        rounds_context = ""
        if previous_rounds:
            rounds_context = "\n\nPrevious rounds:\n"
            for i, round_data in enumerate(previous_rounds[-3:], 1):  # Last 3 rounds
                rounds_context += f"Round {round_data['round']}: Generous={round_data['generous']}, Harsh={round_data['harsh']}\n"
        
        # Generous model's turn to adjust
        generous_prompt = f"""
You are a GENEROUS essay grader negotiating with a harsh grader.

PROMPT: {prompt}
ESSAY: {essay}

Current situation:
- Your grade: {generous_grade}
- Harsh grader's grade: {harsh_grade}
- Round: {round_num}{rounds_context}

As a GENEROUS grader, you want to:
- Stand your ground firmly - you see value in this student's work
- Make only very small concessions if the harsh grader has undeniable points
- Stay true to your generous philosophy - students deserve encouragement
- Move toward the harsh grader's grade only slightly, if at all
- Remember: You are GENEROUS and see the best in students
- Don't give in too easily - your perspective is valuable

IMPORTANT: Your new grade must be LESS generous than your current grade ({generous_grade}). 
You cannot give a higher grade than {generous_grade}.

Consider the harsh grader's perspective but maintain your generous standards.
Provide ONLY a single integer grade from 0-100.
"""
        
        try:
            generous_response = ask(generous_prompt, self.generous_model)
            generous_match = re.search(r'\b(\d+)\b', generous_response)
            new_generous_grade = int(generous_match.group(1)) if generous_match else generous_grade
            new_generous_grade = max(0, min(100, new_generous_grade))
            
            # Ensure the generous grade doesn't get more generous
            if new_generous_grade >= generous_grade:
                print(f"    Warning: Generous model tried to increase grade from {generous_grade} to {new_generous_grade}. Keeping {generous_grade}.")
                new_generous_grade = generous_grade
                
        except Exception as e:
            print(f"Error in generous model haggling: {e}")
            new_generous_grade = generous_grade
        
        # Harsh model's turn to adjust
        harsh_prompt = f"""
You are a HARSH essay grader negotiating with a generous grader.

PROMPT: {prompt}
ESSAY: {essay}

Current situation:
- Generous grader's grade: {new_generous_grade}
- Your grade: {harsh_grade}
- Round: {round_num}{rounds_context}

As a HARSH grader, you want to:
- Stand your ground if you believe your grade is fair
- Make small concessions if the generous grader has valid points
- Stay true to your high standards
- Move toward the generous grader's grade, but not too much
- Remember: You maintain rigorous academic standards
- Don't compromise your standards easily

IMPORTANT: Your new grade must be MORE generous than your current grade ({harsh_grade}). 
You cannot give a lower grade than {harsh_grade}.

Consider the generous grader's perspective but maintain your harsh standards.
Provide ONLY a single integer grade from 0-100.
"""
        
        try:
            harsh_response = ask(harsh_prompt, self.harsh_model)
            harsh_match = re.search(r'\b(\d+)\b', harsh_response)
            new_harsh_grade = int(harsh_match.group(1)) if harsh_match else harsh_grade
            new_harsh_grade = max(0, min(100, new_harsh_grade))
            
            # Ensure the harsh grade doesn't get harsher
            if new_harsh_grade <= harsh_grade:
                print(f"    Warning: Harsh model tried to decrease grade from {harsh_grade} to {new_harsh_grade}. Keeping {harsh_grade}.")
                new_harsh_grade = harsh_grade
                
        except Exception as e:
            print(f"Error in harsh model haggling: {e}")
            new_harsh_grade = harsh_grade
        
        return new_generous_grade, new_harsh_grade
    
    def has_converged(self, generous_grade: int, harsh_grade: int) -> bool:
        """Check if the two grades have converged within threshold"""
        return abs(generous_grade - harsh_grade) <= self.convergence_threshold
    
    def get_final_grade(self, generous_grade: int, harsh_grade: int) -> int:
        """Calculate final consensus grade"""
        # Simple average of the two grades
        return round((generous_grade + harsh_grade) / 2)
    
    def process_essay(self, essay_id: int, prompt: str, essay: str, actual_grade: float) -> Dict:
        """Process a single essay through the haggling system"""
        print(f"Processing essay {essay_id} with haggling system...")
        
        # Convert actual grade from 0-15 to 0-100 scale
        actual_grade_100 = self.convert_grade_scale(actual_grade)
        print(f"  Actual grade: {actual_grade} (0-15 scale) → {actual_grade_100} (0-100 scale)")
        
        # Get baseline grade
        baseline_grade = self.get_baseline_grade(essay, prompt)
        print(f"  Baseline grade: {baseline_grade}")
        
        # Get initial grades from both models
        print("  Getting initial grades...")
        initial_generous = self.get_initial_grade(essay, prompt, self.generous_model, is_generous=True)
        initial_harsh = self.get_initial_grade(essay, prompt, self.harsh_model, is_generous=False)
        
        print(f"    Initial generous grade: {initial_generous}")
        print(f"    Initial harsh grade: {initial_harsh}")
        
        # Check if grades need to be swapped (harsh > generous)
        if initial_harsh > initial_generous:
            print(f"    Warning: Harsh grade ({initial_harsh}) > Generous grade ({initial_generous}). Swapping grades.")
            initial_generous, initial_harsh = initial_harsh, initial_generous
            print(f"    After swap: Generous={initial_generous}, Harsh={initial_harsh}")
        
        # Start haggling
        current_generous = initial_generous
        current_harsh = initial_harsh
        rounds_used = 0
        haggling_history = []
        
        for round_num in range(1, self.max_rounds + 1):
            print(f"  Round {round_num}...")
            
            # Record current state
            haggling_history.append({
                'round': round_num,
                'generous': current_generous,
                'harsh': current_harsh
            })
            
            # Check if we've converged
            if self.has_converged(current_generous, current_harsh):
                rounds_used = round_num
                print(f"    Grades converged! Generous: {current_generous}, Harsh: {current_harsh}")
                break
            
            # One round of haggling
            new_generous, new_harsh = self.haggle_round(
                essay, prompt, current_generous, current_harsh, round_num, haggling_history
            )
            
            print(f"    Generous: {current_generous} → {new_generous}")
            print(f"    Harsh: {current_harsh} → {new_harsh}")
            
            # Check if grades need to be swapped after haggling
            if new_harsh > new_generous:
                print(f"    Warning: Harsh grade ({new_harsh}) > Generous grade ({new_generous}). Swapping grades.")
                new_generous, new_harsh = new_harsh, new_generous
                print(f"    After swap: Generous={new_generous}, Harsh={new_harsh}")
            
            current_generous = new_generous
            current_harsh = new_harsh
            
            # Small delay to avoid rate limiting
            time.sleep(1)
        
        if rounds_used == 0:
            rounds_used = self.max_rounds
            print(f"    Did not converge after {self.max_rounds} rounds")
        
        # Calculate final consensus grade
        final_grade = self.get_final_grade(current_generous, current_harsh)
        print(f"  Final consensus grade: {final_grade}")
        
        return {
            'id': essay_id,
            'baseline_grade': baseline_grade,
            'initial_generous_grade': initial_generous,
            'initial_harsh_grade': initial_harsh,
            'final_generous_grade': current_generous,
            'final_harsh_grade': current_harsh,
            'consensus_grade': final_grade,
            'actual_grade_0_15': actual_grade,
            'actual_grade_0_100': actual_grade_100,
            'rounds_used': rounds_used,
            'converged': self.has_converged(current_generous, current_harsh),
            'grade_difference': abs(current_generous - current_harsh),
            'haggling_history': haggling_history
        }
    
    def run_analysis(self, data_file: str, output_file: str, max_essays: int = None):
        """Run haggling analysis on essays with resume capability"""
        print(f"Loading data from {data_file}...")
        
        # Read TSV data
        df = pd.read_csv(data_file, sep='\t')
        
        if max_essays:
            df = df.head(max_essays)
        
        print(f"Processing {len(df)} essays...")
        
        # Check if we have existing results to resume from
        existing_results = []
        if os.path.exists(output_file):
            print(f"Found existing results file: {output_file}")
            existing_df = pd.read_csv(output_file)
            existing_results = existing_df.to_dict('records')
            print(f"Loaded {len(existing_results)} existing results")
            
            # Get list of already processed essay IDs
            processed_ids = set(result['id'] for result in existing_results)
            print(f"Already processed essays: {sorted(processed_ids)}")
            
            # Filter out already processed essays
            df = df[~df['id'].isin(processed_ids)]
            print(f"Remaining essays to process: {len(df)}")
            
            if len(df) == 0:
                print("All essays have been processed!")
                self.print_summary(existing_results)
                return
        
        results = existing_results.copy()
        
        for idx, row in df.iterrows():
            essay_id = row['id']
            prompt = row['prompt']
            essay = row['essay']
            actual_grade = row['total']
            
            print(f"\n{'='*60}")
            print(f"Processing Essay {essay_id} ({idx+1}/{len(df)})")
            print(f"{'='*60}")
            
            try:
                result = self.process_essay(essay_id, prompt, essay, actual_grade)
                results.append(result)
                
                # Save after each essay for resume capability
                self.save_results(results, output_file)
                print(f"✓ Saved progress after essay {essay_id}")
                    
            except Exception as e:
                print(f"Error processing essay {essay_id}: {e}")
                continue
            
            # Delay between essays
            time.sleep(2)
        
        print(f"\nAnalysis complete! Results saved to {output_file}")
        
        # Print summary statistics
        self.print_summary(results)
    
    def run_all_model_combinations(self, data_file: str, output_file: str, max_essays: int = None):
        """Run haggling analysis with same-model combinations (self-haggling)"""
        print(f"Loading data from {data_file}...")
        
        # Read TSV data
        df = pd.read_csv(data_file, sep='\t')
        
        if max_essays:
            df = df.head(max_essays)
        
        print(f"Processing {len(df)} essays with same-model combinations...")
        print(f"Each essay will be processed 3 times (once with each model as both generous and harsh)")
        
        # Check if we have existing results to resume from
        existing_results = []
        remaining_essays = []
        
        if os.path.exists(output_file):
            print(f"Found existing results file: {output_file}")
            existing_df = pd.read_csv(output_file)
            existing_results = existing_df.to_dict('records')
            print(f"Loaded {len(existing_results)} existing results")
            
            # Get list of already processed essay-model combinations
            processed_combinations = set()
            for result in existing_results:
                combo = (result['id'], result['generous_model'], result['harsh_model'])
                processed_combinations.add(combo)
            
            print(f"Already processed combinations: {len(processed_combinations)}")
            
            # Filter out already processed combinations
            for idx, row in df.iterrows():
                essay_id = row['id']
                for model in self.models:
                    # Same model acting as both generous and harsh (self-haggling)
                    combo = (essay_id, model.name, model.name)
                    if combo not in processed_combinations:
                        remaining_essays.append((idx, row, model, model))
            
            print(f"Remaining essay-model combinations to process: {len(remaining_essays)}")
            
            if len(remaining_essays) == 0:
                print("All essay-model combinations have been processed!")
                self.print_summary_all_combinations(existing_results)
                return
        else:
            # No existing file, create all combinations
            print(f"No existing results file found. Creating same-model combinations...")
            for idx, row in df.iterrows():
                essay_id = row['id']
                for model in self.models:
                    # Same model acting as both generous and harsh (self-haggling)
                    remaining_essays.append((idx, row, model, model))
            
            print(f"Total essay-model combinations to process: {len(remaining_essays)}")
        
        results = existing_results.copy()
        
        # Process remaining combinations
        for i, (idx, row, generous_model, harsh_model) in enumerate(remaining_essays):
            essay_id = row['id']
            prompt = row['prompt']
            essay = row['essay']
            actual_grade = row['total']
            
            print(f"\n{'='*60}")
            print(f"Processing Essay {essay_id} with {generous_model.name} (generous) vs {harsh_model.name} (harsh)")
            print(f"Progress: {i+1}/{len(remaining_essays)}")
            print(f"{'='*60}")
            
            try:
                # Temporarily set the models for this run
                original_generous = self.generous_model
                original_harsh = self.harsh_model
                self.generous_model = generous_model
                self.harsh_model = harsh_model
                
                result = self.process_essay(essay_id, prompt, essay, actual_grade)
                
                # Add model information to result
                result['generous_model'] = generous_model.name
                result['harsh_model'] = harsh_model.name
                
                results.append(result)
                
                # Restore original models
                self.generous_model = original_generous
                self.harsh_model = original_harsh
                
                # Save after each combination for resume capability
                self.save_results_all_combinations(results, output_file)
                print(f"✓ Saved progress after essay {essay_id} with {generous_model.name} vs {harsh_model.name}")
                    
            except Exception as e:
                print(f"Error processing essay {essay_id} with {generous_model.name} vs {harsh_model.name}: {e}")
                continue
            
            # Delay between combinations
            time.sleep(2)
        
        print(f"\nAll model combinations analysis complete! Results saved to {output_file}")
        
        # Print summary statistics
        self.print_summary_all_combinations(results)
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to CSV file"""
        if not results:
            return
        
        # Flatten the results for CSV storage
        flattened_results = []
        for result in results:
            flat_result = {
                'id': result['id'],
                'baseline_grade': result['baseline_grade'],
                'initial_generous_grade': result['initial_generous_grade'],
                'initial_harsh_grade': result['initial_harsh_grade'],
                'final_generous_grade': result['final_generous_grade'],
                'final_harsh_grade': result['final_harsh_grade'],
                'consensus_grade': result['consensus_grade'],
                'actual_grade_0_15': result['actual_grade_0_15'],
                'actual_grade_0_100': result['actual_grade_0_100'],
                'rounds_used': result['rounds_used'],
                'converged': result['converged'],
                'grade_difference': result['grade_difference']
            }
            flattened_results.append(flat_result)
        
        df_results = pd.DataFrame(flattened_results)
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    def save_results_all_combinations(self, results: List[Dict], output_file: str):
        """Save results to CSV file with model combination information"""
        if not results:
            return
        
        # Flatten the results for CSV storage
        flattened_results = []
        for result in results:
            flat_result = {
                'id': result['id'],
                'generous_model': result.get('generous_model', 'Unknown'),
                'harsh_model': result.get('harsh_model', 'Unknown'),
                'baseline_grade': result['baseline_grade'],
                'initial_generous_grade': result['initial_generous_grade'],
                'initial_harsh_grade': result['initial_harsh_grade'],
                'final_generous_grade': result['final_generous_grade'],
                'final_harsh_grade': result['final_harsh_grade'],
                'consensus_grade': result['consensus_grade'],
                'actual_grade_0_15': result['actual_grade_0_15'],
                'actual_grade_0_100': result['actual_grade_0_100'],
                'rounds_used': result['rounds_used'],
                'converged': result['converged'],
                'grade_difference': result['grade_difference']
            }
            flattened_results.append(flat_result)
        
        df_results = pd.DataFrame(flattened_results)
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("HAGGLING SYSTEM SUMMARY STATISTICS")
        print("="*60)
        
        print(f"Essays processed: {len(df)}")
        print(f"Average baseline grade: {df['baseline_grade'].mean():.1f}")
        print(f"Average initial generous grade: {df['initial_generous_grade'].mean():.1f}")
        print(f"Average initial harsh grade: {df['initial_harsh_grade'].mean():.1f}")
        print(f"Average final generous grade: {df['final_generous_grade'].mean():.1f}")
        print(f"Average final harsh grade: {df['final_harsh_grade'].mean():.1f}")
        print(f"Average consensus grade: {df['consensus_grade'].mean():.1f}")
        print(f"Average actual grade (0-100): {df['actual_grade_0_100'].mean():.1f}")
        print(f"Average actual grade (0-15): {df['actual_grade_0_15'].mean():.1f}")
        print(f"Average rounds used: {df['rounds_used'].mean():.1f}")
        print(f"Convergence rate: {df['converged'].mean()*100:.1f}%")
        print(f"Average final grade difference: {df['grade_difference'].mean():.1f}")
        
        # Grade accuracy analysis (using 0-100 scale for comparison)
        baseline_rmse = ((df['baseline_grade'] - df['actual_grade_0_100']) ** 2).mean() ** 0.5
        consensus_rmse = ((df['consensus_grade'] - df['actual_grade_0_100']) ** 2).mean() ** 0.5
        generous_rmse = ((df['initial_generous_grade'] - df['actual_grade_0_100']) ** 2).mean() ** 0.5
        harsh_rmse = ((df['initial_harsh_grade'] - df['actual_grade_0_100']) ** 2).mean() ** 0.5
        
        print(f"\nAccuracy Analysis (0-100 scale):")
        print(f"  Baseline RMSE: {baseline_rmse:.1f}")
        print(f"  Consensus grade RMSE: {consensus_rmse:.1f}")
        print(f"  Initial generous RMSE: {generous_rmse:.1f}")
        print(f"  Initial harsh RMSE: {harsh_rmse:.1f}")
        
        # Show which approach is better
        approaches = [('baseline', baseline_rmse), ('consensus', consensus_rmse), ('generous', generous_rmse), ('harsh', harsh_rmse)]
        best_approach = min(approaches, key=lambda x: x[1])
        print(f"  Best approach: {best_approach[0]} (RMSE: {best_approach[1]:.1f})")
        
        # Improvement analysis
        baseline_improvement = baseline_rmse - consensus_rmse
        print(f"  Haggling improvement over baseline: {baseline_improvement:.1f} RMSE points")
        
        # Additional analysis
        print(f"\nGrade Distribution Analysis:")
        print(f"  Baseline bias: {df['baseline_grade'].mean() - df['actual_grade_0_100'].mean():.1f} points")
        print(f"  Generous bias: {df['initial_generous_grade'].mean() - df['actual_grade_0_100'].mean():.1f} points")
        print(f"  Harsh bias: {df['initial_harsh_grade'].mean() - df['actual_grade_0_100'].mean():.1f} points")
        print(f"  Consensus bias: {df['consensus_grade'].mean() - df['actual_grade_0_100'].mean():.1f} points")
    
    def print_summary_all_combinations(self, results: List[Dict]):
        """Print summary statistics for all model combinations"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("ALL MODEL COMBINATIONS SUMMARY STATISTICS")
        print("="*60)
        
        print(f"Total combinations processed: {len(df)}")
        print(f"Essays processed: {df['id'].nunique()}")
        print(f"Model combinations: {df['generous_model'].nunique()} × {df['harsh_model'].nunique()}")
        
        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"Average baseline grade: {df['baseline_grade'].mean():.1f}")
        print(f"Average initial generous grade: {df['initial_generous_grade'].mean():.1f}")
        print(f"Average initial harsh grade: {df['initial_harsh_grade'].mean():.1f}")
        print(f"Average final generous grade: {df['final_generous_grade'].mean():.1f}")
        print(f"Average final harsh grade: {df['final_harsh_grade'].mean():.1f}")
        print(f"Average consensus grade: {df['consensus_grade'].mean():.1f}")
        print(f"Average actual grade (0-100): {df['actual_grade_0_100'].mean():.1f}")
        print(f"Average rounds used: {df['rounds_used'].mean():.1f}")
        print(f"Convergence rate: {df['converged'].mean()*100:.1f}%")
        print(f"Average final grade difference: {df['grade_difference'].mean():.1f}")
        
        # Model-specific analysis
        print(f"\nModel Performance Analysis:")
        
        for model_name in ['CHATGPT', 'CLAUDE', 'QWEN']:
            print(f"\n{model_name} as Generous Model:")
            generous_data = df[df['generous_model'] == model_name]
            if len(generous_data) > 0:
                print(f"  Average initial grade: {generous_data['initial_generous_grade'].mean():.1f}")
                print(f"  Average final grade: {generous_data['final_generous_grade'].mean():.1f}")
                print(f"  Average bias: {generous_data['initial_generous_grade'].mean() - generous_data['actual_grade_0_100'].mean():.1f}")
                print(f"  RMSE: {((generous_data['initial_generous_grade'] - generous_data['actual_grade_0_100']) ** 2).mean() ** 0.5:.1f}")
            
            print(f"{model_name} as Harsh Model:")
            harsh_data = df[df['harsh_model'] == model_name]
            if len(harsh_data) > 0:
                print(f"  Average initial grade: {harsh_data['initial_harsh_grade'].mean():.1f}")
                print(f"  Average final grade: {harsh_data['final_harsh_grade'].mean():.1f}")
                print(f"  Average bias: {harsh_data['initial_harsh_grade'].mean() - harsh_data['actual_grade_0_100'].mean():.1f}")
                print(f"  RMSE: {((harsh_data['initial_harsh_grade'] - harsh_data['actual_grade_0_100']) ** 2).mean() ** 0.5:.1f}")
        
        # Combination-specific analysis
        print(f"\nModel Combination Analysis:")
        for model_name in ['CHATGPT', 'CLAUDE', 'QWEN']:
            combo_data = df[(df['generous_model'] == model_name) & (df['harsh_model'] == model_name)]
            if len(combo_data) > 0:
                print(f"\n{model_name} (generous) vs {model_name} (harsh) - Self-Haggling:")
                print(f"  Essays processed: {len(combo_data)}")
                print(f"  Average consensus grade: {combo_data['consensus_grade'].mean():.1f}")
                print(f"  Average rounds used: {combo_data['rounds_used'].mean():.1f}")
                print(f"  Convergence rate: {combo_data['converged'].mean()*100:.1f}%")
                print(f"  Consensus RMSE: {((combo_data['consensus_grade'] - combo_data['actual_grade_0_100']) ** 2).mean() ** 0.5:.1f}")
        
        # Overall accuracy analysis
        baseline_rmse = ((df['baseline_grade'] - df['actual_grade_0_100']) ** 2).mean() ** 0.5
        consensus_rmse = ((df['consensus_grade'] - df['actual_grade_0_100']) ** 2).mean() ** 0.5
        
        print(f"\nOverall Accuracy Analysis:")
        print(f"  Baseline RMSE: {baseline_rmse:.1f}")
        print(f"  Consensus RMSE: {consensus_rmse:.1f}")
        print(f"  Improvement: {baseline_rmse - consensus_rmse:.1f} RMSE points")
        
        if consensus_rmse < baseline_rmse:
            improvement_pct = (baseline_rmse - consensus_rmse) / baseline_rmse * 100
            print(f"  Haggling improves over baseline by {improvement_pct:.1f}%")
        else:
            degradation_pct = (consensus_rmse - baseline_rmse) / baseline_rmse * 100
            print(f"  Haggling degrades from baseline by {degradation_pct:.1f}%")

def main():
    """Main function to run the haggling system"""
    print("Haggling Grading System")
    print("="*50)
    
    # Initialize system
    system = HagglingGradingSystem(max_rounds=8, convergence_threshold=3)
    
    # Load the full dataset
    input_file = "DREsS_New.tsv"
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file, sep='\t')
    print(f"Dataset loaded: {len(df)} essays")
    
    # Create a smaller sample for testing
    sample_size = 50  # Start with a smaller sample
    sample_file = f"DREsS_haggling_sample_{sample_size}.tsv"
    
    # Check if sample file already exists
    if os.path.exists(sample_file):
        print(f"Using existing sample file: {sample_file}")
        sample_df = pd.read_csv(sample_file, sep='\t')
    else:
        print(f"Creating new sample of {sample_size} essays...")
        sample_df = df.sample(n=sample_size, random_state=42)
        sample_df.to_csv(sample_file, sep='\t', index=False)
        print(f"Sample saved to {sample_file}")
    
    # Choose analysis mode
    print(f"\nChoose analysis mode:")
    print(f"1. Single model combination (ChatGPT generous vs Claude harsh)")
    print(f"2. Same-model combinations (self-haggling with ChatGPT, Claude, Qwen)")
    
    mode = input("Enter choice (1 or 2): ").strip()
    
    if mode == "2":
        # Run same-model combinations
        output_file = f"haggling_self_combinations_{sample_size}.csv"
        print(f"\nRunning same-model combinations analysis...")
        print(f"Each essay will be processed 3 times (ChatGPT, Claude, Qwen each acting as both generous and harsh)")
        print(f"Results will be saved to: {output_file}")
        print(f"You can stop and restart at any time - the system will resume from where it left off.")
        
        system.run_all_model_combinations(sample_file, output_file)
    else:
        # Run single model combination
        output_file = f"haggling_grading_results_{sample_size}.csv"
        print(f"\nRunning single model combination analysis...")
        print(f"Results will be saved to: {output_file}")
        print(f"You can stop and restart at any time - the system will resume from where it left off.")
        
        system.run_analysis(sample_file, output_file)

if __name__ == "__main__":
    main()
