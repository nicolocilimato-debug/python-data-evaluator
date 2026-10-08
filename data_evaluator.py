import math
from typing import List, Dict, Any

class DataEvaluator:
    """Class for calculating dataset metrics and detecting data anomalies."""

    def __init__(self, data: List[float]):
        self.data = data

    def get_statistics(self) -> Dict[str, float]:
        """Calculates mean, variance, and standard deviation."""
        if not self.data:
            return {}

        n = len(self.data)
        mean_val = sum(self.data) / n
        variance = sum((x - mean_val) ** 2 for x in self.data) / n
        std_dev = math.sqrt(variance)

        return {
            "sample_size": n,
            "mean": round(mean_val, 4),
            "variance": round(variance, 4),
            "std_dev": round(std_dev, 4)
        }

    def filter_outliers(self, max_std_devs: float = 2.0) -> List[float]:
        """Filters out values beyond N standard deviations from the mean."""
        stats = self.get_statistics()
        if not stats:
            return []

        mean, std_dev = stats["mean"], stats["std_dev"]
        return [x for x in self.data if abs(x - mean) <= max_std_devs * std_dev]


def assess_quality(stats: Dict[str, float]) -> str:
    """Evaluates stability based on standard deviation."""
    std = stats.get("std_dev", 0.0)
    if std < 1.0:
        return "High Consistency"
    elif std < 2.5:
        return "Moderate Variance"
    else:
        return "High Volatility"


if __name__ == "__main__":
    dataset = [12.4, 11.8, 12.1, 12.9, 28.5, 12.0, 11.9, 12.3]

    evaluator = DataEvaluator(dataset)
    stats = evaluator.get_statistics()

    print("--- Statistical Summary ---")
    for key, val in stats.items():
        print(f"{key.replace('_', ' ').capitalize()}: {val}")

    print(f"Quality Assessment: {assess_quality(stats)}")
    print(f"Filtered Data (No Outliers): {evaluator.filter_outliers(1.5)}")
