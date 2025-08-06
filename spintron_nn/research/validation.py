"""
Statistical validation and reproducibility framework for spintronic research.

Provides rigorous experimental design, statistical validation tools,
and reproducibility frameworks for academic publication standards.
"""

import time
import json
import numpy as np
import scipy.stats as stats
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import hashlib
import pickle

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments."""
    
    experiment_name: str
    description: str
    random_seed: int
    sample_size: int
    significance_level: float = 0.05
    effect_size_threshold: float = 0.5
    power_threshold: float = 0.8
    replications: int = 3
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass 
class StatisticalResult:
    """Container for statistical analysis results."""
    
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    power: float
    significant: bool
    interpretation: str


@dataclass
class ReproducibilityReport:
    """Comprehensive reproducibility report."""
    
    experiment_config: ExperimentConfig
    results_summary: Dict[str, Any]
    statistical_tests: List[StatisticalResult]
    data_integrity_hash: str
    code_version_hash: str
    environment_info: Dict[str, str]
    reproducibility_score: float
    recommendations: List[str]


class StatisticalValidator:
    """
    Comprehensive statistical validation framework for research experiments.
    
    Provides power analysis, effect size calculations, multiple comparison
    corrections, and publication-ready statistical reporting.
    """
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_history = []
        
        logger.info(f"Initialized StatisticalValidator with output: {output_dir}")
    
    def design_experiment(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        test_type: str = "two_sample"
    ) -> Dict[str, Any]:
        """
        Design experiment with proper power analysis.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power
            alpha: Significance level
            test_type: Type of statistical test
            
        Returns:
            Experiment design recommendations
        """
        
        logger.info(f"Designing experiment - Effect size: {effect_size}, Power: {power}")
        
        # Calculate required sample size
        if test_type == "two_sample":
            # Cohen's d to sample size calculation
            za = stats.norm.ppf(1 - alpha/2)  # Two-tailed
            zb = stats.norm.ppf(power)
            
            n_per_group = 2 * ((za + zb) / effect_size) ** 2
            total_n = int(np.ceil(n_per_group * 2))
            
        elif test_type == "paired":
            za = stats.norm.ppf(1 - alpha/2)
            zb = stats.norm.ppf(power)
            
            total_n = int(np.ceil(((za + zb) / effect_size) ** 2))
            
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Power analysis verification
        actual_power = self._calculate_power(total_n, effect_size, alpha, test_type)
        
        design = {
            "test_type": test_type,
            "target_effect_size": effect_size,
            "target_power": power,
            "significance_level": alpha,
            "recommended_sample_size": total_n,
            "actual_power": actual_power,
            "minimum_detectable_effect": self._minimum_detectable_effect(total_n, power, alpha, test_type),
            "recommendations": self._generate_design_recommendations(total_n, effect_size, power)
        }
        
        logger.info(f"Experiment design completed - Recommended N: {total_n}")
        
        return design
    
    def validate_experiment_results(
        self,
        group1_data: np.ndarray,
        group2_data: np.ndarray,
        experiment_config: ExperimentConfig,
        test_type: str = "automatic"
    ) -> List[StatisticalResult]:
        """
        Comprehensive statistical validation of experimental results.
        
        Args:
            group1_data: First group measurements
            group2_data: Second group measurements  
            experiment_config: Experiment configuration
            test_type: Type of statistical test or 'automatic'
            
        Returns:
            List of statistical test results
        """
        
        logger.info(f"Validating experiment: {experiment_config.experiment_name}")
        
        results = []
        
        # Descriptive statistics
        desc_stats = self._descriptive_statistics(group1_data, group2_data)
        
        # Normality tests
        normality_results = self._test_normality(group1_data, group2_data)
        results.extend(normality_results)
        
        # Homoscedasticity test
        homoscedasticity_result = self._test_homoscedasticity(group1_data, group2_data)
        results.append(homoscedasticity_result)
        
        # Determine appropriate test
        if test_type == "automatic":
            test_type = self._select_appropriate_test(normality_results, homoscedasticity_result)
        
        # Main statistical test
        main_result = self._perform_main_test(group1_data, group2_data, test_type)
        results.append(main_result)
        
        # Effect size calculation
        effect_size_result = self._calculate_effect_size_result(group1_data, group2_data)
        results.append(effect_size_result)
        
        # Power analysis
        power_result = self._post_hoc_power_analysis(group1_data, group2_data, main_result.p_value)
        results.append(power_result)
        
        # Multiple comparison correction if needed
        if len(results) > 1:
            results = self._apply_multiple_comparison_correction(results)
        
        # Save validation results
        self._save_validation_results(results, experiment_config, desc_stats)
        
        logger.info(f"Validation completed - Main test p-value: {main_result.p_value:.4f}")
        
        return results
    
    def meta_analysis(
        self,
        experiment_results: List[Dict[str, Any]],
        outcome_metric: str = "effect_size"
    ) -> Dict[str, Any]:
        """
        Perform meta-analysis across multiple experiments.
        
        Args:
            experiment_results: List of experiment result dictionaries
            outcome_metric: Metric to analyze across experiments
            
        Returns:
            Meta-analysis results
        """
        
        logger.info(f"Performing meta-analysis on {len(experiment_results)} experiments")
        
        # Extract effect sizes and sample sizes
        effect_sizes = []
        sample_sizes = []
        variances = []
        
        for result in experiment_results:
            if outcome_metric in result:
                effect_sizes.append(result[outcome_metric])
                sample_sizes.append(result.get('sample_size', 30))  # Default if missing
                
                # Calculate variance for effect size
                n = result.get('sample_size', 30)
                variance = (n + result[outcome_metric]**2 / (2*n))
                variances.append(variance)
        
        effect_sizes = np.array(effect_sizes)
        variances = np.array(variances)
        weights = 1.0 / variances  # Inverse variance weighting
        
        # Fixed-effects meta-analysis
        weighted_mean_effect = np.sum(weights * effect_sizes) / np.sum(weights)
        se_weighted_mean = 1.0 / np.sqrt(np.sum(weights))
        
        # Confidence interval
        ci_lower = weighted_mean_effect - 1.96 * se_weighted_mean
        ci_upper = weighted_mean_effect + 1.96 * se_weighted_mean
        
        # Heterogeneity assessment (I² statistic)
        Q = np.sum(weights * (effect_sizes - weighted_mean_effect)**2)
        df = len(effect_sizes) - 1
        i_squared = max(0, (Q - df) / Q) if Q > 0 else 0
        
        # Random-effects model if significant heterogeneity
        if i_squared > 0.25:  # Moderate heterogeneity threshold
            tau_squared = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
            re_weights = 1.0 / (variances + tau_squared)
            re_weighted_mean = np.sum(re_weights * effect_sizes) / np.sum(re_weights)
            re_se = 1.0 / np.sqrt(np.sum(re_weights))
        else:
            re_weighted_mean = weighted_mean_effect
            re_se = se_weighted_mean
        
        meta_results = {
            "n_experiments": len(experiment_results),
            "fixed_effects": {
                "pooled_effect_size": weighted_mean_effect,
                "standard_error": se_weighted_mean,
                "confidence_interval": (ci_lower, ci_upper),
                "z_score": weighted_mean_effect / se_weighted_mean,
                "p_value": 2 * (1 - stats.norm.cdf(abs(weighted_mean_effect / se_weighted_mean)))
            },
            "random_effects": {
                "pooled_effect_size": re_weighted_mean,
                "standard_error": re_se,
                "tau_squared": tau_squared if i_squared > 0.25 else 0,
            },
            "heterogeneity": {
                "Q_statistic": Q,
                "df": df,
                "i_squared": i_squared,
                "interpretation": self._interpret_heterogeneity(i_squared)
            },
            "recommendation": "random_effects" if i_squared > 0.25 else "fixed_effects"
        }
        
        # Save meta-analysis results
        output_file = self.output_dir / "meta_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(meta_results, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Meta-analysis completed - Pooled effect size: {weighted_mean_effect:.3f}")
        
        return meta_results
    
    def _calculate_power(
        self,
        sample_size: int,
        effect_size: float,
        alpha: float,
        test_type: str
    ) -> float:
        """Calculate statistical power for given parameters."""
        
        if test_type == "two_sample":
            n_per_group = sample_size // 2
            se = np.sqrt(2 / n_per_group)
        else:  # paired
            se = 1 / np.sqrt(sample_size)
        
        za = stats.norm.ppf(1 - alpha/2)
        z_beta = (effect_size / se) - za
        power = stats.norm.cdf(z_beta)
        
        return max(0.0, min(1.0, power))
    
    def _minimum_detectable_effect(
        self,
        sample_size: int,
        power: float,
        alpha: float,
        test_type: str
    ) -> float:
        """Calculate minimum detectable effect size."""
        
        za = stats.norm.ppf(1 - alpha/2)
        zb = stats.norm.ppf(power)
        
        if test_type == "two_sample":
            n_per_group = sample_size // 2
            mde = (za + zb) * np.sqrt(2 / n_per_group)
        else:  # paired
            mde = (za + zb) / np.sqrt(sample_size)
        
        return mde
    
    def _generate_design_recommendations(
        self,
        sample_size: int,
        effect_size: float,
        power: float
    ) -> List[str]:
        """Generate experiment design recommendations."""
        
        recommendations = []
        
        if sample_size > 1000:
            recommendations.append("Large sample size detected - consider computational constraints")
        
        if effect_size < 0.2:
            recommendations.append("Small effect size - ensure measurement precision and control confounds")
        
        if power < 0.8:
            recommendations.append("Power below recommended threshold - consider increasing sample size")
        
        recommendations.append("Use randomization for group assignment")
        recommendations.append("Consider blocking on important covariates")
        recommendations.append("Plan for potential dropout/missing data")
        
        return recommendations
    
    def _descriptive_statistics(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics."""
        
        stats_dict = {
            "group1": {
                "n": len(group1),
                "mean": np.mean(group1),
                "std": np.std(group1, ddof=1),
                "median": np.median(group1),
                "q25": np.percentile(group1, 25),
                "q75": np.percentile(group1, 75),
                "min": np.min(group1),
                "max": np.max(group1),
                "skewness": stats.skew(group1),
                "kurtosis": stats.kurtosis(group1)
            },
            "group2": {
                "n": len(group2),
                "mean": np.mean(group2),
                "std": np.std(group2, ddof=1),
                "median": np.median(group2),
                "q25": np.percentile(group2, 25),
                "q75": np.percentile(group2, 75),
                "min": np.min(group2),
                "max": np.max(group2),
                "skewness": stats.skew(group2),
                "kurtosis": stats.kurtosis(group2)
            }
        }
        
        return stats_dict
    
    def _test_normality(self, group1: np.ndarray, group2: np.ndarray) -> List[StatisticalResult]:
        """Test normality assumptions."""
        
        results = []
        
        # Shapiro-Wilk test for each group
        for i, group in enumerate([group1, group2], 1):
            if len(group) <= 5000:  # Shapiro-Wilk limitation
                stat, p_val = stats.shapiro(group)
                result = StatisticalResult(
                    test_name=f"Shapiro-Wilk Normality Test - Group {i}",
                    statistic=stat,
                    p_value=p_val,
                    effect_size=0.0,  # Not applicable
                    confidence_interval=(0.0, 0.0),
                    power=0.0,  # Not applicable
                    significant=p_val < 0.05,
                    interpretation="Normal distribution" if p_val >= 0.05 else "Non-normal distribution"
                )
            else:
                # Use Anderson-Darling for large samples
                result_ad = stats.anderson(group)
                result = StatisticalResult(
                    test_name=f"Anderson-Darling Normality Test - Group {i}",
                    statistic=result_ad.statistic,
                    p_value=0.05,  # Approximate
                    effect_size=0.0,
                    confidence_interval=(0.0, 0.0),
                    power=0.0,
                    significant=result_ad.statistic > result_ad.critical_values[2],  # 5% level
                    interpretation="Normal distribution" if result_ad.statistic <= result_ad.critical_values[2] else "Non-normal distribution"
                )
            
            results.append(result)
        
        return results
    
    def _test_homoscedasticity(self, group1: np.ndarray, group2: np.ndarray) -> StatisticalResult:
        """Test homoscedasticity (equal variances)."""
        
        # Levene's test
        stat, p_val = stats.levene(group1, group2)
        
        result = StatisticalResult(
            test_name="Levene's Test for Equal Variances",
            statistic=stat,
            p_value=p_val,
            effect_size=0.0,  # Not applicable
            confidence_interval=(0.0, 0.0),
            power=0.0,  # Not applicable
            significant=p_val < 0.05,
            interpretation="Equal variances" if p_val >= 0.05 else "Unequal variances"
        )
        
        return result
    
    def _select_appropriate_test(
        self,
        normality_results: List[StatisticalResult],
        homoscedasticity_result: StatisticalResult
    ) -> str:
        """Select appropriate statistical test based on assumptions."""
        
        # Check if both groups are normal
        both_normal = all(not result.significant for result in normality_results)
        equal_variances = not homoscedasticity_result.significant
        
        if both_normal and equal_variances:
            return "t_test"
        elif both_normal and not equal_variances:
            return "welch_t_test"
        else:
            return "mann_whitney"
    
    def _perform_main_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        test_type: str
    ) -> StatisticalResult:
        """Perform main statistical test."""
        
        if test_type == "t_test":
            stat, p_val = stats.ttest_ind(group1, group2)
            test_name = "Independent t-test"
        elif test_type == "welch_t_test":
            stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
            test_name = "Welch's t-test"
        elif test_type == "mann_whitney":
            stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Calculate confidence interval for difference in means
        mean_diff = np.mean(group1) - np.mean(group2)
        se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
        
        if test_type in ["t_test", "welch_t_test"]:
            df = len(group1) + len(group2) - 2 if test_type == "t_test" else None
            t_crit = stats.t.ppf(0.975, df) if df else 1.96
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff
        else:  # Mann-Whitney
            # Approximate CI for Mann-Whitney
            ci_lower = mean_diff - 1.96 * se_diff
            ci_upper = mean_diff + 1.96 * se_diff
        
        result = StatisticalResult(
            test_name=test_name,
            statistic=stat,
            p_value=p_val,
            effect_size=0.0,  # Will be calculated separately
            confidence_interval=(ci_lower, ci_upper),
            power=0.0,  # Will be calculated separately
            significant=p_val < 0.05,
            interpretation="Significant difference" if p_val < 0.05 else "No significant difference"
        )
        
        return result
    
    def _calculate_effect_size_result(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> StatisticalResult:
        """Calculate effect size (Cohen's d)."""
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Confidence interval for Cohen's d
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
        ci_lower = cohens_d - 1.96 * se_d
        ci_upper = cohens_d + 1.96 * se_d
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            interpretation = "Negligible effect"
        elif abs(cohens_d) < 0.5:
            interpretation = "Small effect"
        elif abs(cohens_d) < 0.8:
            interpretation = "Medium effect"
        else:
            interpretation = "Large effect"
        
        result = StatisticalResult(
            test_name="Cohen's d Effect Size",
            statistic=cohens_d,
            p_value=0.0,  # Not applicable
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            power=0.0,  # Not applicable
            significant=abs(cohens_d) >= 0.2,  # Arbitrary threshold
            interpretation=interpretation
        )
        
        return result
    
    def _post_hoc_power_analysis(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        p_value: float
    ) -> StatisticalResult:
        """Perform post-hoc power analysis."""
        
        # Calculate observed effect size
        mean1, mean2 = np.mean(group1), np.mean(group2)
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        observed_effect_size = abs(mean1 - mean2) / pooled_std
        
        # Calculate power
        n_per_group = min(len(group1), len(group2))
        power = self._calculate_power(n_per_group * 2, observed_effect_size, 0.05, "two_sample")
        
        # Interpretation
        if power >= 0.8:
            interpretation = "Adequate power"
        elif power >= 0.5:
            interpretation = "Moderate power"
        else:
            interpretation = "Insufficient power"
        
        result = StatisticalResult(
            test_name="Post-hoc Power Analysis",
            statistic=power,
            p_value=p_value,
            effect_size=observed_effect_size,
            confidence_interval=(0.0, 1.0),  # Power is between 0 and 1
            power=power,
            significant=power >= 0.8,
            interpretation=interpretation
        )
        
        return result
    
    def _apply_multiple_comparison_correction(
        self,
        results: List[StatisticalResult]
    ) -> List[StatisticalResult]:
        """Apply multiple comparison correction."""
        
        # Extract p-values from results that have meaningful p-values
        p_values = []
        result_indices = []
        
        for i, result in enumerate(results):
            if result.test_name in ["Independent t-test", "Welch's t-test", "Mann-Whitney U test"]:
                p_values.append(result.p_value)
                result_indices.append(i)
        
        if len(p_values) > 1:
            # Bonferroni correction
            corrected_alpha = 0.05 / len(p_values)
            
            # Update significance based on corrected alpha
            for i, result_idx in enumerate(result_indices):
                original_result = results[result_idx]
                corrected_significant = p_values[i] < corrected_alpha
                
                # Create corrected result
                corrected_result = StatisticalResult(
                    test_name=original_result.test_name + " (Bonferroni corrected)",
                    statistic=original_result.statistic,
                    p_value=original_result.p_value,
                    effect_size=original_result.effect_size,
                    confidence_interval=original_result.confidence_interval,
                    power=original_result.power,
                    significant=corrected_significant,
                    interpretation="Significant after correction" if corrected_significant else "Not significant after correction"
                )
                
                results[result_idx] = corrected_result
        
        return results
    
    def _save_validation_results(
        self,
        results: List[StatisticalResult],
        config: ExperimentConfig,
        descriptive_stats: Dict[str, Any]
    ):
        """Save validation results to file."""
        
        output_data = {
            "experiment_config": asdict(config),
            "descriptive_statistics": descriptive_stats,
            "statistical_tests": [asdict(result) for result in results],
            "timestamp": datetime.now().isoformat()
        }
        
        output_file = self.output_dir / f"validation_{config.experiment_name}_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Validation results saved to: {output_file}")
    
    def _interpret_heterogeneity(self, i_squared: float) -> str:
        """Interpret I² heterogeneity statistic."""
        
        if i_squared < 0.25:
            return "Low heterogeneity"
        elif i_squared < 0.5:
            return "Moderate heterogeneity"  
        elif i_squared < 0.75:
            return "Substantial heterogeneity"
        else:
            return "Considerable heterogeneity"


class ReproducibilityFramework:
    """
    Framework for ensuring research reproducibility.
    
    Provides experiment tracking, data integrity verification,
    code versioning, and environment documentation.
    """
    
    def __init__(self, project_name: str, output_dir: str = "reproducibility"):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_registry = {}
        self.data_registry = {}
        
        logger.info(f"Initialized ReproducibilityFramework for: {project_name}")
    
    def register_experiment(
        self,
        experiment_config: ExperimentConfig,
        code_snapshot: Optional[str] = None,
        data_files: Optional[List[str]] = None
    ) -> str:
        """
        Register an experiment for reproducibility tracking.
        
        Args:
            experiment_config: Experiment configuration
            code_snapshot: Optional code snapshot or git commit hash
            data_files: List of data file paths
            
        Returns:
            Unique experiment ID
        """
        
        # Generate unique experiment ID
        experiment_id = hashlib.md5(
            f"{experiment_config.experiment_name}_{experiment_config.timestamp}".encode()
        ).hexdigest()[:8]
        
        # Environment information
        environment_info = self._capture_environment_info()
        
        # Data integrity hashes
        data_hashes = {}
        if data_files:
            for data_file in data_files:
                if Path(data_file).exists():
                    data_hashes[data_file] = self._calculate_file_hash(data_file)
        
        # Code version hash
        code_hash = code_snapshot or self._calculate_code_hash()
        
        experiment_record = {
            "experiment_id": experiment_id,
            "config": asdict(experiment_config),
            "environment_info": environment_info,
            "data_hashes": data_hashes,
            "code_hash": code_hash,
            "registration_timestamp": datetime.now().isoformat()
        }
        
        self.experiment_registry[experiment_id] = experiment_record
        
        # Save registry
        self._save_experiment_registry()
        
        logger.info(f"Registered experiment: {experiment_id}")
        
        return experiment_id
    
    def validate_reproducibility(
        self,
        experiment_id: str,
        results_data: Dict[str, Any],
        tolerance: float = 0.01
    ) -> ReproducibilityReport:
        """
        Validate reproducibility of an experiment.
        
        Args:
            experiment_id: Experiment identifier
            results_data: Current experiment results
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Comprehensive reproducibility report
        """
        
        if experiment_id not in self.experiment_registry:
            raise ValueError(f"Experiment {experiment_id} not found in registry")
        
        original_record = self.experiment_registry[experiment_id]
        original_config = ExperimentConfig(**original_record["config"])
        
        # Current environment info
        current_env = self._capture_environment_info()
        
        # Check data integrity
        data_integrity_check = self._verify_data_integrity(
            original_record["data_hashes"]
        )
        
        # Check code version
        current_code_hash = self._calculate_code_hash()
        code_version_match = current_code_hash == original_record["code_hash"]
        
        # Environment comparison
        env_compatibility = self._compare_environments(
            original_record["environment_info"], current_env
        )
        
        # Calculate reproducibility score
        reproducibility_score = self._calculate_reproducibility_score(
            data_integrity_check, code_version_match, env_compatibility
        )
        
        # Generate recommendations
        recommendations = self._generate_reproducibility_recommendations(
            data_integrity_check, code_version_match, env_compatibility
        )
        
        # Create comprehensive report
        report = ReproducibilityReport(
            experiment_config=original_config,
            results_summary=results_data,
            statistical_tests=[],  # To be populated if available
            data_integrity_hash=str(data_integrity_check),
            code_version_hash=current_code_hash,
            environment_info=current_env,
            reproducibility_score=reproducibility_score,
            recommendations=recommendations
        )
        
        # Save reproducibility report
        self._save_reproducibility_report(report, experiment_id)
        
        logger.info(f"Reproducibility validation completed - Score: {reproducibility_score:.2f}")
        
        return report
    
    def _capture_environment_info(self) -> Dict[str, str]:
        """Capture current environment information."""
        
        import sys
        import platform
        
        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        }
        
        # Add package versions if available
        try:
            import torch
            env_info["torch_version"] = torch.__version__
        except ImportError:
            pass
        
        try:
            import numpy
            env_info["numpy_version"] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import scipy
            env_info["scipy_version"] = scipy.__version__
        except ImportError:
            pass
        
        return env_info
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _calculate_code_hash(self) -> str:
        """Calculate hash of current code state."""
        
        # Simple implementation - in practice, would use git commit hash
        # or comprehensive directory hashing
        current_time = datetime.now().isoformat()
        hash_input = f"code_snapshot_{current_time}".encode()
        
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def _verify_data_integrity(self, original_hashes: Dict[str, str]) -> bool:
        """Verify data file integrity."""
        
        for file_path, original_hash in original_hashes.items():
            if not Path(file_path).exists():
                return False
            
            current_hash = self._calculate_file_hash(file_path)
            if current_hash != original_hash:
                return False
        
        return True
    
    def _compare_environments(
        self,
        original_env: Dict[str, str],
        current_env: Dict[str, str]
    ) -> float:
        """Compare environment compatibility."""
        
        compatibility_score = 0.0
        total_components = len(original_env)
        
        for key, original_value in original_env.items():
            if key in current_env:
                if current_env[key] == original_value:
                    compatibility_score += 1.0
                else:
                    # Partial credit for compatible versions
                    if "version" in key.lower():
                        # Simple version compatibility check
                        compatibility_score += 0.5
                    else:
                        compatibility_score += 0.0
            else:
                compatibility_score += 0.0
        
        return compatibility_score / total_components if total_components > 0 else 1.0
    
    def _calculate_reproducibility_score(
        self,
        data_integrity: bool,
        code_match: bool,
        env_compatibility: float
    ) -> float:
        """Calculate overall reproducibility score."""
        
        # Weighted average of components
        weights = {
            "data_integrity": 0.4,
            "code_match": 0.3,
            "env_compatibility": 0.3
        }
        
        score = (
            weights["data_integrity"] * int(data_integrity) +
            weights["code_match"] * int(code_match) +
            weights["env_compatibility"] * env_compatibility
        )
        
        return score
    
    def _generate_reproducibility_recommendations(
        self,
        data_integrity: bool,
        code_match: bool,
        env_compatibility: float
    ) -> List[str]:
        """Generate recommendations for improving reproducibility."""
        
        recommendations = []
        
        if not data_integrity:
            recommendations.append("Data files have changed - verify data integrity")
        
        if not code_match:
            recommendations.append("Code version differs - document changes or revert to original")
        
        if env_compatibility < 0.9:
            recommendations.append("Environment differences detected - use containerization")
        
        recommendations.extend([
            "Set random seeds for all random operations",
            "Document all software versions and dependencies",
            "Use version control for code and configuration",
            "Save complete analysis pipelines and scripts",
            "Document hardware specifications if relevant"
        ])
        
        return recommendations
    
    def _save_experiment_registry(self):
        """Save experiment registry to file."""
        
        registry_file = self.output_dir / "experiment_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self.experiment_registry, f, indent=2)
    
    def _save_reproducibility_report(self, report: ReproducibilityReport, experiment_id: str):
        """Save reproducibility report to file."""
        
        report_file = self.output_dir / f"reproducibility_report_{experiment_id}.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Reproducibility report saved to: {report_file}")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy data types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)