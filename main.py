# src/main.py
from typing import Dict, Any, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from detectors import SCurveDetector, ExponentialDetector, LinearDetector, DecayDetector
from transformers import LogitTransformer, LogTransformer, IdentityTransformer


class FeatureAnalyser:
    def __init__(self, correlation_threshold = 0.7):
        self.correlation_threshold = correlation_threshold
        self.detectors = {
            'linear': LinearDetector(),
            'S_curve': SCurveDetector(),
            'exponential': ExponentialDetector(),
            'exponential_decay': DecayDetector(),
        }
        self.analysis_results = {}
        self.feature_patterns = {}
        self.transformers = {}
    
    def analyze_dataset(self, df, target_col):
        # Analyse all features wrt to taget for all pattern combinations

        # get numerical column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        for feature in numeric_cols:
            # print(feature)
            result = self._analyse_feature_pattern(
                df[feature], df[target_col]
            )
            # print(result)

            self.analysis_results[feature] = result

            # get max score
            # print(result)
            if result:
                best_pattern = max(result.items(), key=lambda x: x[1]['score'])
                self.feature_patterns[feature] = {
                    'pattern_type': best_pattern[0],
                    # 'correlation': best_pattern[1]['correlation'],
                    'score': best_pattern[1]['score'],
                    'params': best_pattern[1]['params']
                }
        
        # print("**",best_pattern)

        return self.analysis_results

    def _analyse_feature_pattern(self, x, y):
        # try all patterns for [x, y]

        # correlation check
        correlation = np.corrcoef(x, y)[0, 1]

        pattern_results = {}
        # pattern_results['correlation'] = correlation
        
        for pattern_name, detector in self.detectors.items():
            # detector is class and "detector" -> objest will store results when we fit data into it.
            try:
                current_pattern = detector.detect(x, y)
                # print("*******",current_pattern)
                if current_pattern:
                    score = detector.get_fit_score()

                    pattern_results[pattern_name] = {
                        'score': score,
                        'params': detector.get_parameters()
                    }
            except Exception as e:
                print(f"Error detecting {pattern_name} pattern: {str(e)}")
        
        return pattern_results

    def get_recommended_transformations(self):
        transformations = {}

        for feature, pattern_info in self.feature_patterns.items():
            pattern_type = pattern_info['pattern_type']
            score = pattern_info['score']

            if score < 0.6:
                # print(f"Warning: No clear pattern detected for {feature}.")
                warnings.warn(f"Warning: No clear pattern detected for {feature}.")
            
                transformations[feature] = 'none'
            else:
                transform_map = {
                    'linear': 'none',
                    'exponential': 'log',
                    'exponential_decay': 'log',
                    'S_curve': 'logit'
                }
                print(f"Recommending: {pattern_type}")
                transformations[feature] = transform_map.get(pattern_type, 'none')
        
        return transformations
    
    def transform_features(self, df):
        df_transformed = df.copy()
        transformations = self.get_recommended_transformations()

        for feature, transform_type in transformations.items():
            if feature in df.columns:
                try:
                    transformer = self._get_transformer(transform_type)
                    df_transformed[feature] = transformer.fit_transform(
                        df[feature].values.reshape(-1, 1)
                    ).ravel()
                    print(f"Applied {transform_type} transformation to {feature}")
                except Exception as e:
                    warnings.warn(f"Could not transform {feature}: {str(e)}")
        
        return df_transformed

    def _get_transformer(self, transform_type):
        
        if transform_type == 'boxcox':
            return PowerTransformer(method='box-cox')
        elif transform_type == 'log':
            return LogTransformer()
        elif transform_type == 'logit':
            return LogitTransformer()
        else:
            return IdentityTransformer()

    def get_pattern_summary(self, feature_name):
        if feature_name not in self.analysis_results.keys():
            return f"No analysis found for {feature_name}"

        result = self.analysis_results[feature_name]
        # print(result)
        best_pattern = self.feature_patterns.get(feature_name, {})

        summary = [
            f"Feature: {feature_name}",
            # f"\nCorrelation with target: {result['correlation']:.3f}"
        ]
        if best_pattern:
            summary.extend([
                "\nBest Pattern:",
                f"  Type: {best_pattern['pattern_type']}",
                f"  Score: {best_pattern['score']:.3f}"
            ])
        
        # dist = result['distribution']
        # summary.extend([
        #     "\nDistribution Statistics:",
        #     f"  Skewness: {dist['skew']:.3f}",
        #     f"  Kurtosis: {dist['kurtosis']:.3f}"
        # ])

        transformations = self.get_recommended_transformations()
        if feature_name in transformations:
            summary.append(f"\nRecommended transformation: {transformations[feature_name]}")

        return "\n".join(summary)

# Plot comparison
def plot_comparison(df_original, df_transformed):
    # Define features and their labels
    features = {
        'linear_var': 'Linear Variable',
        'quadratic_var': 'Quadratic Variable',
        'exponential_var': 'Exponential Variable',
        'logarithmic_var': 'Logarithmic Variable',
        's_curve_var': 'S-Curve Variable',
        'decay_var': 'Decay Variable'
    }
    
    # Create subplot grid
    fig, axes = plt.subplots(len(features), 2, figsize=(15, 20))
    fig.suptitle('Feature Transformation Comparison', fontsize=16, y=1.02)
    
    # Plot each feature
    for idx, (feature, label) in enumerate(features.items()):
        # Original data
        axes[idx, 0].scatter(df_original['target'], df_original[feature], 
                           alpha=0.5, color='blue')
        axes[idx, 0].set_title(f'Original {label}')
        axes[idx, 0].set_xlabel('Target')
        axes[idx, 0].set_ylabel(label)
        
        # Transformed data
        axes[idx, 1].scatter(df_transformed['target'], df_transformed[feature], 
                           alpha=0.5, color='green')
        axes[idx, 1].set_title(f'Transformed {label}')
        axes[idx, 1].set_xlabel('Target')
        axes[idx, 1].set_ylabel(f'Transformed {label}')
    
    # Adjust layout
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Create dataset
    np.random.seed(42)
    n_samples = 1000
    
    # linear_data = np.random.normal(0, 1, n_samples)
    # target_data = .8 * linear_data + np.random.normal(0, 0.5, n_samples)
    # df = pd.DataFrame({
    #     'linear_var': linear_data,
    #     'exp_var': np.exp(np.random.normal(0, 0.5, n_samples)),
    #     'decay_var': np.random.exponential(2, n_samples),
    #     'target': target_data
    # })

    # Generate independent variables with different relationships
    # df = pd.DataFrame({
    #     # 'linear_var': np.random.normal(0, 1, n_samples),  
    #     # 'quadratic_var': np.random.normal(0, 1, n_samples)**2,  
    #     # 'exponential_var': np.exp(np.random.normal(0, 1, n_samples)), 
    #     'exponential_var': np.exp(np.random.uniform(-2, 5, n_samples)), 
    #     # 'logarithmic_var': np.log(np.abs(np.random.normal(0, 1, n_samples)) + 1),  
    #     # 'random_var': np.random.normal(0, 1, n_samples)  # Random noise feature
    # })

    # # Define a single target variable using a weighted combination of all features
    # df['target'] = (
    #     # 0.1 * df['linear_var'] + 
    #     0.2 * np.sqrt(df['quadratic_var']) +
    #     0.8 * np.log(df['exponential_var'] + 1)
    #     # np.random.normal(0, 0.5, n_samples)  # Adding some noise
    # )

    df = pd.DataFrame({
        'target': np.linspace(1, 100, n_samples)
    })

    df['linear_var'] = 0.5 * df['target'] + 0.02 * np.random.normal(0, 5, n_samples)  # Linear relationship
    df['quadratic_var'] = 0.01 * df['target']**2 + 0.02 * np.random.normal(0, 5, n_samples)  # Quadratic pattern
    df['exponential_var'] = np.exp(df['target'] / 20) + 0.02 * np.random.normal(0, 0.5, n_samples)  # Exponential
    df['logarithmic_var'] = np.log(df['target'] + 1) + 0.02 * np.random.normal(0, 0.2, n_samples)  # Logarithmic
    df['s_curve_var'] = 1 / (1 + np.exp(-0.1 * (df['target'] - 50))) + 0.02 * np.random.normal(0, 0.02, n_samples)  # S-curve
    df['decay_var'] = np.exp(-df['target'] / 30) + 0.02 * np.random.normal(0, 0.02, n_samples)  # Decay


    # for feature in df.columns:
    #     plot_and_analyze(feature, 'target', df)

    analyzer = FeatureAnalyser(correlation_threshold=0.7)
    results = analyzer.analyze_dataset(df, 'target')
    print("*")
    print(analyzer.feature_patterns)
    print("*")

    # # Print pattern analysis for each feature
    # for feature in df.columns:
    #     if feature != 'target':
    #         print("\n" + "="*50)
    #         print(analyzer.get_pattern_summary(feature))

    # Transform features based on analysed patttern
    df_transformed = analyzer.transform_features(df)

    print("FEAUTRES TARANFORMED")
    # plot_transformed_relations(df, df_transformed, target_var="target")
    # Create and show plots
    fig = plot_comparison(df, df_transformed)
    plt.savefig("plot_comparision.png")
    plt.show()



    # Print correlation improvements
    print("\n" + "="*50)
    print("\nCorrelations with target before transformation:")
    print(df.corr()['target'])
    print("\n" + "="*50)
    print("\nCorrelations with target after transformation:")
    print(df_transformed.corr()['target'])