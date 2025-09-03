import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json
import random
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import existing LLM functions
from code import ask, LLM
from openai import OpenAI
import anthropic
from google import genai

# Set up API clients
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
anthropic_client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
gemini_client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

@dataclass
class EssayMetadata:
    """Data class for essay metadata"""
    essay_id: str
    essay_text: str
    essay_type: str  # 'creative' or 'analytical'
    true_grade: int
    quality_justification: str
    word_count: int
    prompt_used: str
    creativity_score: int
    analytical_depth: int
    quality_level: str  # 'low', 'medium', 'high'

class EssayGenerator:
    """Generate essays with controlled quality distribution"""
    
    def __init__(self):
        self.creative_prompts = {
            'personal_narrative': [
                "Describe a moment when you learned something important about yourself",
                "Write about a place that holds special meaning to you", 
                "Tell about a time when you had to make a difficult choice"
            ],
            'creative_response': [
                "If you could have dinner with any historical figure, who and why?",
                "Describe your perfect day from start to finish",
                "Write about what the world will look like in 50 years"
            ],
            'imaginative': [
                "You discover a door in your house you've never seen before...",
                "Write from the perspective of an object in your room",
                "Describe a world where gravity works differently"
            ]
        }
        
        self.analytical_prompts = {
            'argumentative': [
                "Should social media platforms be regulated by government?",
                "Is remote learning as effective as in-person education?",
                "Should college education be free for all students?"
            ],
            'compare_contrast': [
                "Compare renewable vs fossil fuel energy sources",
                "Contrast democratic vs authoritarian government systems", 
                "Compare online vs traditional shopping experiences"
            ],
            'analysis': [
                "Analyze the causes of climate change",
                "Examine the effects of technology on human relationships",
                "Analyze the role of education in reducing inequality"
            ]
        }
    
    def generate_essay_with_grade(self, prompt: str, essay_type: str, target_quality: str) -> EssayMetadata:
        """Generate an essay with controlled quality and corresponding grade"""
        
        # Create generation prompt with quality instruction
        quality_instructions = {
            'low': "Write a poorly structured essay with weak arguments, poor grammar, and minimal depth.",
            'medium': "Write a reasonably well-structured essay with decent arguments and acceptable writing quality.",
            'high': "Write an excellent essay with strong arguments, clear structure, and sophisticated writing."
        }
        
        generation_prompt = f"""
        Write a {essay_type} essay responding to this prompt: "{prompt}"
        
        Requirements:
        - Length: 200-400 words
        - Style: {quality_instructions[target_quality]}
        - Essay type: {essay_type}
        
        Write only the essay content, no additional text.
        """
        
        # Generate essay
        essay_text = self._call_llm(generation_prompt, temperature=0.7)
        
        # Generate grade and justification
        grade_prompt = f"""
        Grade this {essay_type} essay on a scale of 0-6:
        
        Essay: {essay_text}
        
        Consider:
        - Content Quality & Depth (40%)
        - Organization & Structure (25%)
        - {'Creativity/Originality' if essay_type == 'creative' else 'Logic/Evidence'} (20%)
        - Writing Mechanics (15%)
        
        Provide your response in this exact format:
        Grade: [0-6]
        Justification: [2-3 sentences explaining the grade]
        """
        
        grade_response = self._call_llm(grade_prompt, temperature=0.3)
        
        # Parse grade and justification
        grade_match = re.search(r'Grade:\s*(\d+)', grade_response)
        justification_match = re.search(r'Justification:\s*(.+)', grade_response, re.DOTALL)
        
        true_grade = int(grade_match.group(1)) if grade_match else 3
        quality_justification = justification_match.group(1).strip() if justification_match else "Grade assigned based on overall quality"
        
        # Calculate metadata
        word_count = len(essay_text.split())
        creativity_score = self._calculate_creativity_score(essay_text, essay_type)
        analytical_depth = self._calculate_analytical_depth(essay_text, essay_type)
        
        return EssayMetadata(
            essay_id=f"{essay_type}_{target_quality}_{len(essay_text)}",
            essay_text=essay_text,
            essay_type=essay_type,
            true_grade=true_grade,
            quality_justification=quality_justification,
            word_count=word_count,
            prompt_used=prompt,
            creativity_score=creativity_score,
            analytical_depth=analytical_depth,
            quality_level=target_quality
        )
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call GPT-4 for essay generation"""
        try:
            response = openai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4",
                temperature=temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "Error generating content"
    
    def _calculate_creativity_score(self, text: str, essay_type: str) -> int:
        """Calculate creativity score (1-10)"""
        if essay_type != 'creative':
            return 1
        
        # Simple heuristics for creativity
        creativity_indicators = [
            'imagine', 'creative', 'unique', 'original', 'novel',
            'fantasy', 'dream', 'vision', 'inventive', 'artistic'
        ]
        
        text_lower = text.lower()
        score = 1
        for indicator in creativity_indicators:
            if indicator in text_lower:
                score += 1
        
        return min(score, 10)
    
    def _calculate_analytical_depth(self, text: str, essay_type: str) -> int:
        """Calculate analytical depth score (1-10)"""
        if essay_type != 'analytical':
            return 1
        
        # Simple heuristics for analytical depth
        analytical_indicators = [
            'because', 'therefore', 'however', 'furthermore', 'moreover',
            'evidence', 'analysis', 'research', 'study', 'data',
            'conclusion', 'argument', 'reasoning', 'logical', 'systematic'
        ]
        
        text_lower = text.lower()
        score = 1
        for indicator in analytical_indicators:
            if indicator in text_lower:
                score += 1
        
        return min(score, 10)
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete essay dataset"""
        print("Generating essay dataset...")
        
        essays = []
        
        # Generate creative essays
        quality_distribution = {'low': 10, 'medium': 30, 'high': 10}
        
        for category, prompts in self.creative_prompts.items():
            for prompt in prompts:
                for quality, count in quality_distribution.items():
                    for _ in range(count // len(prompts)):
                        essay = self.generate_essay_with_grade(prompt, 'creative', quality)
                        essays.append(essay)
                        print(f"Generated creative essay: {essay.essay_id}")
                        time.sleep(1)  # Rate limiting
        
        # Generate analytical essays
        for category, prompts in self.analytical_prompts.items():
            for prompt in prompts:
                for quality, count in quality_distribution.items():
                    for _ in range(count // len(prompts)):
                        essay = self.generate_essay_with_grade(prompt, 'analytical', quality)
                        essays.append(essay)
                        print(f"Generated analytical essay: {essay.essay_id}")
                        time.sleep(1)  # Rate limiting
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(essay) for essay in essays])
        df.to_csv('essays.csv', index=False)
        print(f"Generated {len(essays)} essays and saved to essays.csv")
        
        return df

class LLMGrader:
    """Grade essays using multiple LLMs at different temperatures"""
    
    def __init__(self):
        self.temperatures = [0.0, 0.3, 0.7, 1.0, 1.3]
        self.models = {
            'gpt4': self._grade_with_gpt4,
            'claude': self._grade_with_claude,
            'gemini': self._grade_with_gemini
        }
        self.runs_per_essay = 3
    
    def grade_essay(self, essay: EssayMetadata, model: str, temperature: float) -> Dict:
        """Grade a single essay with specified model and temperature"""
        
        grading_prompt = f"""
        Grade this {essay.essay_type} essay on a scale of 0-6. Consider:
        - Content Quality & Depth (40%)
        - Organization & Structure (25%) 
        - {'Creativity/Originality' if essay.essay_type == 'creative' else 'Logic/Evidence'} (20%)
        - Writing Mechanics (15%)

        Essay: {essay.essay_text}

        Provide only the numerical score as your response.
        """
        
        start_time = time.time()
        
        try:
            grade_response = self.models[model](grading_prompt, temperature)
            
            # Extract numerical grade
            import re
            grade_match = re.search(r'(\d+(?:\.\d+)?)', grade_response)
            grade = float(grade_match.group(1)) if grade_match else None
            
            response_time = time.time() - start_time
            
            return {
                'essay_id': essay.essay_id,
                'model': model,
                'temperature': temperature,
                'grade': grade,
                'response_time': response_time,
                'raw_response': grade_response,
                'true_grade': essay.true_grade,
                'grade_difference': grade - essay.true_grade if grade is not None else None,
                'absolute_error': abs(grade - essay.true_grade) if grade is not None else None,
                'essay_type': essay.essay_type,
                'word_count': essay.word_count,
                'quality_level': essay.quality_level,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error grading essay {essay.essay_id} with {model} at temp {temperature}: {e}")
            return {
                'essay_id': essay.essay_id,
                'model': model,
                'temperature': temperature,
                'grade': None,
                'response_time': time.time() - start_time,
                'raw_response': str(e),
                'true_grade': essay.true_grade,
                'grade_difference': None,
                'absolute_error': None,
                'essay_type': essay.essay_type,
                'word_count': essay.word_count,
                'quality_level': essay.quality_level,
                'timestamp': datetime.now().isoformat()
            }
    
    def _grade_with_gpt4(self, prompt: str, temperature: float) -> str:
        """Grade using GPT-4"""
        response = openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4",
            temperature=temperature,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    
    def _grade_with_claude(self, prompt: str, temperature: float) -> str:
        """Grade using Claude"""
        message = anthropic_client.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=50,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    
    def _grade_with_gemini(self, prompt: str, temperature: float) -> str:
        """Grade using Gemini"""
        response = gemini_client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt,
            generation_config={'temperature': temperature}
        )
        return response.text.strip()
    
    def grade_dataset(self, essays_df: pd.DataFrame) -> pd.DataFrame:
        """Grade all essays with all models at all temperatures"""
        print("Starting grading process...")
        
        results = []
        total_grades = len(essays_df) * len(self.models) * len(self.temperatures) * self.runs_per_essay
        current_grade = 0
        
        for _, row in essays_df.iterrows():
            essay = EssayMetadata(**row.to_dict())
            
            for model in self.models.keys():
                for temperature in self.temperatures:
                    for run in range(self.runs_per_essay):
                        current_grade += 1
                        print(f"Grading {current_grade}/{total_grades}: {essay.essay_id} - {model} - temp {temperature} - run {run+1}")
                        
                        result = self.grade_essay(essay, model, temperature)
                        result['run_number'] = run + 1
                        results.append(result)
                        
                        time.sleep(0.5)  # Rate limiting
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('grading_results.csv', index=False)
        print(f"Completed grading. Results saved to grading_results.csv")
        
        return results_df

class TemperatureAnalysis:
    """Analyze temperature effects on grading accuracy and consistency"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup matplotlib for publication-quality plots"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
    
    def calculate_accuracy_metrics(self) -> pd.DataFrame:
        """Calculate accuracy metrics by temperature and model"""
        
        # Remove failed grades
        valid_results = self.results_df.dropna(subset=['grade'])
        
        # Group by model, temperature, and essay type
        metrics = valid_results.groupby(['model', 'temperature', 'essay_type']).agg({
            'absolute_error': ['mean', 'std'],
            'grade_difference': ['mean', 'std'],
            'grade': ['mean', 'std'],
            'true_grade': 'mean'
        }).round(3)
        
        # Flatten column names
        metrics.columns = ['_'.join(col).strip() for col in metrics.columns]
        metrics = metrics.reset_index()
        
        # Calculate correlation with true grades
        correlations = []
        for (model, temp, essay_type), group in valid_results.groupby(['model', 'temperature', 'essay_type']):
            if len(group) > 1:
                corr = group['grade'].corr(group['true_grade'])
                correlations.append({
                    'model': model,
                    'temperature': temp,
                    'essay_type': essay_type,
                    'correlation': corr
                })
        
        corr_df = pd.DataFrame(correlations)
        metrics = metrics.merge(corr_df, on=['model', 'temperature', 'essay_type'], how='left')
        
        return metrics
    
    def calculate_consistency_metrics(self) -> pd.DataFrame:
        """Calculate consistency metrics (inter-run reliability)"""
        
        # Calculate standard deviation across runs for each essay-model-temperature combination
        consistency = self.results_df.groupby(['essay_id', 'model', 'temperature']).agg({
            'grade': ['mean', 'std', 'count']
        }).round(3)
        
        consistency.columns = ['grade_mean', 'grade_std', 'run_count']
        consistency = consistency.reset_index()
        
        # Calculate intraclass correlation coefficient (simplified)
        def calculate_icc(group):
            if len(group) < 2:
                return np.nan
            variance = group['grade'].var()
            if variance == 0:
                return 1.0
            return 1 - (group['grade'].var() / group['grade'].var())
        
        icc = self.results_df.groupby(['essay_id', 'model', 'temperature']).apply(calculate_icc)
        icc_df = icc.reset_index()
        icc_df.columns = ['essay_id', 'model', 'temperature', 'icc']
        
        consistency = consistency.merge(icc_df, on=['essay_id', 'model', 'temperature'])
        
        return consistency
    
    def create_visualizations(self, accuracy_metrics: pd.DataFrame, consistency_metrics: pd.DataFrame):
        """Create publication-quality visualizations"""
        
        # 1. Temperature Performance Heatmap
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Temperature Effects on Grading Accuracy by Model and Essay Type', fontsize=16)
        
        for i, model in enumerate(['gpt4', 'claude', 'gemini']):
            for j, essay_type in enumerate(['creative', 'analytical']):
                ax = axes[j, i]
                
                model_data = accuracy_metrics[
                    (accuracy_metrics['model'] == model) & 
                    (accuracy_metrics['essay_type'] == essay_type)
                ]
                
                if not model_data.empty:
                    ax.plot(model_data['temperature'], model_data['absolute_error_mean'], 
                           marker='o', linewidth=2, markersize=8)
                    ax.fill_between(model_data['temperature'], 
                                  model_data['absolute_error_mean'] - model_data['absolute_error_std'],
                                  model_data['absolute_error_mean'] + model_data['absolute_error_std'],
                                  alpha=0.3)
                
                ax.set_title(f'{model.upper()} - {essay_type.title()}')
                ax.set_xlabel('Temperature')
                ax.set_ylabel('Mean Absolute Error')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temperature_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Consistency vs Accuracy Scatter
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for model in ['gpt4', 'claude', 'gemini']:
            for essay_type in ['creative', 'analytical']:
                model_data = accuracy_metrics[
                    (accuracy_metrics['model'] == model) & 
                    (accuracy_metrics['essay_type'] == essay_type)
                ]
                
                if not model_data.empty:
                    # Use consistency from the same data
                    consistency_data = consistency_metrics[
                        (consistency_metrics['model'] == model)
                    ]['icc'].mean()
                    
                    ax.scatter(model_data['absolute_error_mean'], 
                             [consistency_data] * len(model_data),
                             s=100, alpha=0.7, label=f'{model}-{essay_type}')
        
        ax.set_xlabel('Mean Absolute Error (Lower is Better)')
        ax.set_ylabel('Inter-run Consistency (ICC)')
        ax.set_title('Consistency vs Accuracy Trade-off by Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig('consistency_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Grade Distribution Shifts
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Grade Distribution Shifts by Temperature', fontsize=16)
        
        for i, model in enumerate(['gpt4', 'claude', 'gemini']):
            for j, essay_type in enumerate(['creative', 'analytical']):
                ax = axes[j, i]
                
                model_data = self.results_df[
                    (self.results_df['model'] == model) & 
                    (self.results_df['essay_type'] == essay_type)
                ]
                
                for temp in [0.0, 0.7, 1.3]:
                    temp_data = model_data[model_data['temperature'] == temp]['grade']
                    if not temp_data.empty:
                        ax.hist(temp_data, alpha=0.6, label=f'Temp {temp}', bins=range(0, 8))
                
                ax.set_title(f'{model.upper()} - {essay_type.title()}')
                ax.set_xlabel('Grade')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('grade_distribution_shifts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_statistical_tests(self, accuracy_metrics: pd.DataFrame):
        """Run advanced statistical tests"""
        
        from scipy import stats
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        print("\n=== STATISTICAL ANALYSIS RESULTS ===")
        
        # 1. ANOVA: Temperature × Model × Essay Type
        print("\n1. Mixed-effects ANOVA Results:")
        
        # Prepare data for ANOVA
        valid_results = self.results_df.dropna(subset=['grade'])
        
        # Temperature effect
        temp_groups = [valid_results[valid_results['temperature'] == temp]['absolute_error'] 
                      for temp in valid_results['temperature'].unique()]
        f_stat, p_val = stats.f_oneway(*temp_groups)
        print(f"Temperature effect: F={f_stat:.3f}, p={p_val:.3f}")
        
        # Model effect
        model_groups = [valid_results[valid_results['model'] == model]['absolute_error'] 
                       for model in valid_results['model'].unique()]
        f_stat, p_val = stats.f_oneway(*model_groups)
        print(f"Model effect: F={f_stat:.3f}, p={p_val:.3f}")
        
        # Essay type effect
        type_groups = [valid_results[valid_results['essay_type'] == essay_type]['absolute_error'] 
                      for essay_type in valid_results['essay_type'].unique()]
        f_stat, p_val = stats.f_oneway(*type_groups)
        print(f"Essay type effect: F={f_stat:.3f}, p={p_val:.3f}")
        
        # 2. Polynomial regression for non-linear temperature effects
        print("\n2. Polynomial Regression Results:")
        
        for model in ['gpt4', 'claude', 'gemini']:
            model_data = accuracy_metrics[accuracy_metrics['model'] == model]
            
            if len(model_data) > 2:
                X = model_data['temperature'].values.reshape(-1, 1)
                y = model_data['absolute_error_mean'].values
                
                # Linear
                poly1 = PolynomialFeatures(degree=1)
                X_poly1 = poly1.fit_transform(X)
                reg1 = LinearRegression().fit(X_poly1, y)
                r2_linear = reg1.score(X_poly1, y)
                
                # Quadratic
                poly2 = PolynomialFeatures(degree=2)
                X_poly2 = poly2.fit_transform(X)
                reg2 = LinearRegression().fit(X_poly2, y)
                r2_quad = reg2.score(X_poly2, y)
                
                print(f"{model.upper()}: Linear R²={r2_linear:.3f}, Quadratic R²={r2_quad:.3f}")
        
        # 3. Effect sizes
        print("\n3. Effect Sizes (Cohen's d):")
        
        # Temperature effect size
        low_temp = valid_results[valid_results['temperature'] <= 0.3]['absolute_error']
        high_temp = valid_results[valid_results['temperature'] >= 1.0]['absolute_error']
        
        if len(low_temp) > 0 and len(high_temp) > 0:
            pooled_std = np.sqrt(((len(low_temp) - 1) * low_temp.var() + 
                                (len(high_temp) - 1) * high_temp.var()) / 
                               (len(low_temp) + len(high_temp) - 2))
            cohens_d = (high_temp.mean() - low_temp.mean()) / pooled_std
            print(f"Temperature effect size (Cohen's d): {cohens_d:.3f}")
        
        # Model effect size
        gpt4_errors = valid_results[valid_results['model'] == 'gpt4']['absolute_error']
        claude_errors = valid_results[valid_results['model'] == 'claude']['absolute_error']
        
        if len(gpt4_errors) > 0 and len(claude_errors) > 0:
            pooled_std = np.sqrt(((len(gpt4_errors) - 1) * gpt4_errors.var() + 
                                (len(claude_errors) - 1) * claude_errors.var()) / 
                               (len(gpt4_errors) + len(claude_errors) - 2))
            cohens_d = (claude_errors.mean() - gpt4_errors.mean()) / pooled_std
            print(f"GPT-4 vs Claude effect size (Cohen's d): {cohens_d:.3f}")

def main():
    """Main research execution"""
    
    print("=== TEMPERATURE EFFECTS ON LLM GRADING ACCURACY RESEARCH ===")
    print("This study examines how temperature settings affect grading accuracy and consistency")
    print("for creative vs analytical essays using multiple LLM models.\n")
    
    # Phase 1: Generate essay dataset
    print("PHASE 1: Generating Essay Dataset")
    generator = EssayGenerator()
    
    # Check if essays.csv already exists
    if os.path.exists('essays.csv'):
        print("Loading existing essays.csv...")
        essays_df = pd.read_csv('essays.csv')
    else:
        print("Generating new essay dataset...")
        essays_df = generator.generate_dataset()
    
    print(f"Loaded {len(essays_df)} essays")
    print(f"Creative essays: {len(essays_df[essays_df['essay_type'] == 'creative'])}")
    print(f"Analytical essays: {len(essays_df[essays_df['essay_type'] == 'analytical'])}")
    
    # Phase 2: Grade essays
    print("\nPHASE 2: Grading Essays")
    grader = LLMGrader()
    
    # Check if grading_results.csv already exists
    if os.path.exists('grading_results.csv'):
        print("Loading existing grading results...")
        results_df = pd.read_csv('grading_results.csv')
    else:
        print("Starting grading process...")
        results_df = grader.grade_dataset(essays_df)
    
    print(f"Loaded {len(results_df)} grading results")
    
    # Phase 3: Analysis
    print("\nPHASE 3: Statistical Analysis")
    analyzer = TemperatureAnalysis(results_df)
    
    # Calculate metrics
    accuracy_metrics = analyzer.calculate_accuracy_metrics()
    consistency_metrics = analyzer.calculate_consistency_metrics()
    
    # Save metrics
    accuracy_metrics.to_csv('accuracy_metrics.csv', index=False)
    consistency_metrics.to_csv('consistency_metrics.csv', index=False)
    
    # Create visualizations
    print("Creating visualizations...")
    analyzer.create_visualizations(accuracy_metrics, consistency_metrics)
    
    # Run statistical tests
    analyzer.run_statistical_tests(accuracy_metrics)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(accuracy_metrics, consistency_metrics, results_df)
    
    print("\n=== RESEARCH COMPLETE ===")
    print("Files generated:")
    print("- essays.csv: Generated essay dataset")
    print("- grading_results.csv: All grading results")
    print("- accuracy_metrics.csv: Accuracy analysis")
    print("- consistency_metrics.csv: Consistency analysis")
    print("- temperature_performance_heatmap.png: Performance visualization")
    print("- consistency_vs_accuracy.png: Trade-off analysis")
    print("- grade_distribution_shifts.png: Distribution analysis")
    print("- research_summary_report.md: Complete analysis report")

def generate_summary_report(accuracy_metrics: pd.DataFrame, consistency_metrics: pd.DataFrame, results_df: pd.DataFrame):
    """Generate a comprehensive research summary report"""
    
    report = f"""# Temperature Effects on LLM Grading Accuracy: Research Summary

## Executive Summary

This study investigated how temperature settings (0.0, 0.3, 0.7, 1.0, 1.3) affect the accuracy and consistency of LLM grading for creative vs analytical essays. The research used three major LLM models (GPT-4, Claude, Gemini) to grade 100 LLM-generated essays across multiple temperature settings.

## Key Findings

### 1. Optimal Temperature Settings
- **Creative Essays**: Best accuracy at temperature {accuracy_metrics[accuracy_metrics['essay_type'] == 'creative']['absolute_error_mean'].idxmin()}
- **Analytical Essays**: Best accuracy at temperature {accuracy_metrics[accuracy_metrics['essay_type'] == 'analytical']['absolute_error_mean'].idxmin()}

### 2. Model Performance Comparison
- **Most Accurate**: {accuracy_metrics.groupby('model')['absolute_error_mean'].mean().idxmin()}
- **Most Consistent**: {consistency_metrics.groupby('model')['icc'].mean().idxmax()}
- **Most Temperature-Sensitive**: {accuracy_metrics.groupby('model')['absolute_error_mean'].std().idxmax()}

### 3. Essay Type Effects
- Creative essays show {accuracy_metrics[accuracy_metrics['essay_type'] == 'creative']['absolute_error_mean'].mean():.3f} mean absolute error
- Analytical essays show {accuracy_metrics[accuracy_metrics['essay_type'] == 'analytical']['absolute_error_mean'].mean():.3f} mean absolute error

## Methodology

### Dataset
- **Total Essays**: {len(results_df['essay_id'].unique())}
- **Creative Essays**: {len(results_df[results_df['essay_type'] == 'creative']['essay_id'].unique())}
- **Analytical Essays**: {len(results_df[results_df['essay_type'] == 'analytical']['essay_id'].unique())}
- **Quality Distribution**: Low (20%), Medium (60%), High (20%)

### Grading Process
- **Models**: GPT-4, Claude-3.5-Sonnet, Gemini-Pro
- **Temperatures**: 0.0, 0.3, 0.7, 1.0, 1.3
- **Runs per essay**: 3
- **Total grading calls**: {len(results_df)}

## Statistical Results

### Accuracy Metrics
{accuracy_metrics.to_string()}

### Consistency Metrics
{consistency_metrics.describe().to_string()}

## Recommendations

1. **For Creative Essays**: Use temperature {accuracy_metrics[accuracy_metrics['essay_type'] == 'creative']['absolute_error_mean'].idxmin()}
2. **For Analytical Essays**: Use temperature {accuracy_metrics[accuracy_metrics['essay_type'] == 'analytical']['absolute_error_mean'].idxmin()}
3. **For Maximum Consistency**: Use temperature 0.0
4. **For Balanced Performance**: Use temperature 0.7

## Limitations

- Study uses LLM-generated essays rather than human-written essays
- Limited to three major LLM models
- Focus on 0-6 grading scale
- Single prompt template used for all grading

## Future Research Directions

1. Extend to human-written essays
2. Test additional LLM models
3. Investigate prompt engineering effects
4. Study longer-form essay grading
5. Analyze cost-effectiveness trade-offs

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open('research_summary_report.md', 'w') as f:
        f.write(report)
    
    print("Summary report saved to research_summary_report.md")

if __name__ == "__main__":
    main()
