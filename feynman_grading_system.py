import pandas as pd
import re
import csv
from typing import Dict, List, Tuple
import time
from code import ask, LLM

def get_representative_sample(df: pd.DataFrame, sample_size: int = 250) -> pd.DataFrame:
    """
    Get a stratified sample based on grade distribution to ensure representativeness.
    
    Args:
        df: Input dataframe with essays
        sample_size: Target number of essays to sample
        
    Returns:
        Stratified sample dataframe
    """
    print(f"Creating stratified sample of {sample_size} essays from {len(df)} total essays...")
    
    # Create grade categories for stratification
    # Assuming 'total' column contains grades 0-15
    df['grade_category'] = pd.cut(df['total'], 
                                 bins=[0, 5, 10, 15], 
                                 labels=['Low (0-5)', 'Medium (6-10)', 'High (11-15)'],
                                 include_lowest=True)
    
    # Show distribution of original dataset
    print("\nOriginal dataset grade distribution:")
    grade_dist = df['grade_category'].value_counts().sort_index()
    for category, count in grade_dist.items():
        print(f"  {category}: {count} essays ({count/len(df)*100:.1f}%)")
    
    # Calculate proportional sample size for each category
    sample_per_category = {}
    for category in df['grade_category'].unique():
        if pd.isna(category):
            continue
        category_count = len(df[df['grade_category'] == category])
        proportional_size = int(sample_size * category_count / len(df))
        sample_per_category[category] = min(proportional_size, category_count)
    
    # Sample from each category
    sample_dfs = []
    for category, target_size in sample_per_category.items():
        if target_size > 0:
            category_df = df[df['grade_category'] == category]
            category_sample = category_df.sample(n=target_size, random_state=42)  # Fixed seed for reproducibility
            sample_dfs.append(category_sample)
            print(f"  Sampled {len(category_sample)} essays from {category}")
    
    # Combine samples
    sample = pd.concat(sample_dfs, ignore_index=True)
    
    # If we need more essays to reach target, fill randomly from remaining
    if len(sample) < sample_size:
        remaining = df[~df.index.isin(sample.index)]
        additional_needed = sample_size - len(sample)
        if len(remaining) >= additional_needed:
            additional = remaining.sample(n=additional_needed, random_state=42)
            sample = pd.concat([sample, additional], ignore_index=True)
            print(f"  Added {len(additional)} additional essays randomly to reach target")
        else:
            print(f"  Warning: Could only sample {len(sample)} essays due to dataset constraints")
    
    # Show final sample distribution
    print(f"\nFinal sample grade distribution:")
    final_dist = sample['grade_category'].value_counts().sort_index()
    for category, count in final_dist.items():
        print(f"  {category}: {count} essays ({count/len(sample)*100:.1f}%)")
    
    # Remove the temporary grade_category column
    sample = sample.drop('grade_category', axis=1)
    
    print(f"\nStratified sample created: {len(sample)} essays")
    return sample

class ReadabilityScorer:
    """Calculate Flesch-Kincaid and Dale-Chall readability scores"""
    
    @staticmethod
    def count_syllables(text: str) -> int:
        """Count syllables in text using simple heuristic"""
        text = text.lower()
        text = re.sub(r'[^a-z]', '', text)
        if len(text) <= 3:
            return 1
        
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        if text.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        return count
    
    @staticmethod
    def count_sentences(text: str) -> int:
        """Count sentences in text"""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        words = re.findall(r'\b\w+\b', text)
        return len(words)
    
    @staticmethod
    def flesch_kincaid(text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level"""
        words = ReadabilityScorer.count_words(text)
        sentences = ReadabilityScorer.count_sentences(text)
        syllables = ReadabilityScorer.count_syllables(text)
        
        if words == 0 or sentences == 0:
            return 0.0
        
        # Flesch-Kincaid formula
        score = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        return round(score, 2)
    
    @staticmethod
    def dale_chall(text: str) -> float:
        """Calculate Dale-Chall Readability Score"""
        words = ReadabilityScorer.count_words(text)
        sentences = ReadabilityScorer.count_sentences(text)
        
        if words == 0 or sentences == 0:
            return 0.0
        
        # List of common words (simplified - in practice this would be much longer)
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
        }
        
        difficult_words = 0
        word_list = re.findall(r'\b\w+\b', text.lower())
        
        for word in word_list:
            if word not in common_words and len(word) > 2:
                difficult_words += 1
        
        # Dale-Chall formula
        score = 0.1579 * (difficult_words / words * 100) + 0.0496 * (words / sentences)
        
        if difficult_words / words > 0.05:
            score += 3.6365
            
        return round(score, 2)

class FeynmanGradingSystem:
    """Main system for implementing Feynman technique with LLM grading"""
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.scorer = ReadabilityScorer()
        
        # Readability threshold (aiming for ~7 year old level)
        # Flesch-Kincaid: Lower grade level = easier to read (aiming for grade 2-3)
        self.flesch_threshold = 3.0  # Target grade level
        
    def get_baseline_grade(self, essay: str, prompt: str, llm: LLM) -> float:
        """Get baseline grade from LLM"""
        grading_prompt = f"""
You are an expert essay grader. Please grade the following essay based on the given prompt.

PROMPT: {prompt}

ESSAY: {essay}

Please provide a single numerical grade from 0-15, where:
- 0-5: Poor (significant issues with content, organization, or language)
- 6-10: Fair (some good points but several areas need improvement)
- 11-15: Good to Excellent (well-developed arguments, good organization, clear language)

Provide ONLY the numerical grade, no explanation.
"""
        
        try:
            response = ask(grading_prompt, llm)
            # Extract numerical grade from response
            grade_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
            if grade_match:
                return float(grade_match.group(1))
            else:
                print(f"Warning: Could not extract grade from response: {response}")
                return 0.0
        except Exception as e:
            print(f"Error getting baseline grade from {llm.name}: {e}")
            return 0.0
    
    def get_core_concept(self, essay: str, prompt: str, llm: LLM) -> str:
        """Extract the core concept from the essay"""
        concept_prompt = f"""
Extract the core concept or main argument from this essay in 1-2 sentences.

PROMPT: {prompt}
ESSAY: {essay}

Core concept:
"""
        
        try:
            response = ask(concept_prompt, llm)
            return response.strip()
        except Exception as e:
            print(f"Error extracting core concept from {llm.name}: {e}")
            return essay[:100] + "..."  # Fallback to first 100 chars
    
    def simplify_explanation(self, concept: str, iteration: int, llm: LLM) -> str:
        """Simplify explanation using Feynman technique - lowering target age"""
        if iteration == 1:
            prompt = f"""
Explain this concept so that a younger child could understand it:

{concept}

Make it appropriate for a younger audience:
"""
        else:
            prompt = f"""
Here's a previous explanation:

{concept}

Now make this explanation appropriate for an even younger child. Lower the target age:
"""
        
        try:
            response = ask(prompt, llm)
            return response.strip()
        except Exception as e:
            print(f"Error simplifying explanation from {llm.name}: {e}")
            return concept  # Fallback to previous version
    
    def meets_readability_thresholds(self, text: str) -> Tuple[bool, float, float]:
        """Check if text meets readability threshold (Flesch-Kincaid only)"""
        flesch_score = self.scorer.flesch_kincaid(text)
        dale_chall_score = self.scorer.dale_chall(text)
        
        # Only check Flesch-Kincaid: lower grade level = easier to read
        meets_threshold = flesch_score <= self.flesch_threshold
        
        return (meets_threshold, flesch_score, dale_chall_score)
    
    def get_final_grade(self, essay: str, prompt: str, simplified_concept: str, llm: LLM) -> float:
        """Get final grade after Feynman explanation"""
        final_prompt = f"""
You are an expert essay grader. Please grade the following essay based on the given prompt.

PROMPT: {prompt}

ESSAY: {essay}

IMPORTANT CONTEXT: The core concept of this essay is: {simplified_concept}

Please provide a single numerical grade from 0-15, where:
- 0-5: Poor (significant issues with content, organization, or language)
- 6-10: Fair (some good points but several areas need improvement)
- 11-15: Good to Excellent (well-developed arguments, good organization, clear language)

Provide ONLY the numerical grade, no explanation.
"""
        
        try:
            response = ask(final_prompt, llm)
            # Extract numerical grade from response
            grade_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
            if grade_match:
                return float(grade_match.group(1))
            else:
                print(f"Warning: Could not extract final grade from response: {response}")
                return 0.0
        except Exception as e:
            print(f"Error getting final grade from {llm.name}: {e}")
            return 0.0
    
    def process_essay(self, essay_id: int, prompt: str, essay: str, actual_grade: float, llm: LLM) -> Dict:
        """Process a single essay through the complete system"""
        print(f"Processing essay {essay_id} with {llm.name}...")
        
        # Get baseline grade
        baseline_grade = self.get_baseline_grade(essay, prompt, llm)
        print(f"  Baseline grade: {baseline_grade}")
        
        # Extract core concept
        core_concept = self.get_core_concept(essay, prompt, llm)
        print(f"  Core concept extracted")
        
        # Iterative Feynman simplification
        current_explanation = core_concept
        final_flesch = 0.0
        final_dale_chall = 0.0
        iterations_used = 0
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"  Iteration {iteration}...")
            
            # Simplify explanation
            current_explanation = self.simplify_explanation(current_explanation, iteration, llm)
            
            # Check readability
            meets_threshold, flesch_score, dale_chall_score = self.meets_readability_thresholds(current_explanation)
            
            print(f"    Flesch-Kincaid: {flesch_score} (target: ≤{self.flesch_threshold}), Dale-Chall: {dale_chall_score}")
            
            if meets_threshold:
                final_flesch = flesch_score
                final_dale_chall = dale_chall_score
                iterations_used = iteration
                print(f"    Target age level reached after {iteration} iterations!")
                break
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        if iterations_used == 0:
            # If we didn't meet target age level, use the last scores
            final_flesch = flesch_score
            final_dale_chall = dale_chall_score
            iterations_used = self.max_iterations
            print(f"    Did not reach target age level after {self.max_iterations} iterations")
        
        # Get final grade with simplified concept
        final_grade = self.get_final_grade(essay, prompt, current_explanation, llm)
        print(f"  Final grade: {final_grade}")
        
        return {
            'id': essay_id,
            'llm': llm.name,
            'baseline_grade': baseline_grade,
            'final_grade': final_grade,
            'actual_grade': actual_grade,
            'final_flesch_kincaid': final_flesch,
            'final_dale_chall': final_dale_chall,
            'iterations_used': iterations_used,
            'core_concept': core_concept,
            'final_explanation': current_explanation
        }
    
    def run_complete_analysis(self, data_file: str, output_file: str, max_essays: int = None):
        """Run complete analysis on all essays with all LLMs"""
        print(f"Loading data from {data_file}...")
        
        # Read TSV data
        df = pd.read_csv(data_file, sep='\t')
        
        if max_essays:
            df = df.head(max_essays)
        
        print(f"Processing {len(df)} essays...")
        
        results = []
        llms_to_test = [LLM.CHATGPT, LLM.CLAUDE, LLM.GEMINI]
        
        for idx, row in df.iterrows():
            essay_id = row['id']
            prompt = row['prompt']
            essay = row['essay']
            actual_grade = row['total']
            
            print(f"\n{'='*60}")
            print(f"Processing Essay {essay_id}")
            print(f"{'='*60}")
            
            for llm in llms_to_test:
                try:
                    result = self.process_essay(essay_id, prompt, essay, actual_grade, llm)
                    results.append(result)
                    
                    # Save intermediate results
                    if len(results) % 3 == 0:  # Save every 3 results
                        self.save_results(results, output_file)
                        
                except Exception as e:
                    print(f"Error processing essay {essay_id} with {llm.name}: {e}")
                    continue
                
                # Delay between LLMs to avoid rate limiting
                time.sleep(1)
            
            # Delay between essays
            time.sleep(2)
        
        # Save final results
        self.save_results(results, output_file)
        print(f"\nAnalysis complete! Results saved to {output_file}")
        
        # Print summary statistics
        self.print_summary(results)
    
    def run_complete_analysis_from_dataframe(self, df: pd.DataFrame, output_file: str):
        """Run complete analysis on essays from a dataframe (for stratified sampling)"""
        print(f"Processing {len(df)} essays from dataframe...")
        
        results = []
        llms_to_test = [LLM.CHATGPT, LLM.CLAUDE, LLM.GEMINI]
        
        for idx, row in df.iterrows():
            essay_id = row['id']
            prompt = row['prompt']
            essay = row['essay']
            actual_grade = row['total']
            
            print(f"\n{'='*60}")
            print(f"Processing Essay {essay_id}")
            print(f"{'='*60}")
            
            for llm in llms_to_test:
                try:
                    result = self.process_essay(essay_id, prompt, essay, actual_grade, llm)
                    results.append(result)
                    
                    # Save intermediate results
                    if len(results) % 3 == 0:  # Save every 3 results
                        self.save_results(results, output_file)
                        
                except Exception as e:
                    print(f"Error processing essay {essay_id} with {llm.name}: {e}")
                    continue
                
                # Delay between LLMs to avoid rate limiting
                time.sleep(1)
            
            # Delay between essays
            time.sleep(2)
        
        # Save final results
        self.save_results(results, output_file)
        print(f"\nAnalysis complete! Results saved to {output_file}")
        
        # Print summary statistics
        self.print_summary(results)
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to CSV file"""
        if not results:
            return
            
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        if not results:
            return
            
        df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        for llm_name in df['llm'].unique():
            llm_data = df[df['llm'] == llm_name]
            
            print(f"\n{llm_name}:")
            print(f"  Essays processed: {len(llm_data)}")
            print(f"  Average baseline grade: {llm_data['baseline_grade'].mean():.2f}")
            print(f"  Average final grade: {llm_data['final_grade'].mean():.2f}")
            print(f"  Average actual grade: {llm_data['actual_grade'].mean():.2f}")
            print(f"  Average Flesch-Kincaid: {llm_data['final_flesch_kincaid'].mean():.2f}")
            print(f"  Average Dale-Chall: {llm_data['final_dale_chall'].mean():.2f}")
            print(f"  Average iterations: {llm_data['iterations_used'].mean():.2f}")
            
            # Grade improvement analysis
            baseline_rmse = ((llm_data['baseline_grade'] - llm_data['actual_grade']) ** 2).mean() ** 0.5
            final_rmse = ((llm_data['final_grade'] - llm_data['actual_grade']) ** 2).mean() ** 0.5
            
            print(f"  Baseline RMSE: {baseline_rmse:.2f}")
            print(f"  Final RMSE: {final_rmse:.2f}")
            print(f"  RMSE Improvement: {baseline_rmse - final_rmse:.2f}")

def main():
    """Main function to run the system"""
    print("Feynman Grading System")
    print("="*50)
    
    # Initialize system with reduced iterations for efficiency
    system = FeynmanGradingSystem(max_iterations=5)
    
    # Load the full dataset first
    input_file = "DREsS_New.tsv"
    print(f"Loading full dataset from {input_file}...")
    df = pd.read_csv(input_file, sep='\t')
    print(f"Full dataset loaded: {len(df)} essays")
    
    # Create stratified sample
    sample_size = 250  # Adjust this number as needed
    sample_df = get_representative_sample(df, sample_size)
    
    # Save the sample for reproducibility and future use
    sample_file = f"DREsS_stratified_sample_{sample_size}.tsv"
    sample_df.to_csv(sample_file, sep='\t', index=False)
    print(f"Stratified sample saved to {sample_file}")
    
    # Run analysis on the stratified sample
    output_file = f"feynman_grading_results_{sample_size}.csv"
    print(f"\nRunning analysis on stratified sample...")
    
    # Use the sample dataframe directly instead of reading from file
    system.run_complete_analysis_from_dataframe(sample_df, output_file)

if __name__ == "__main__":
    main()
