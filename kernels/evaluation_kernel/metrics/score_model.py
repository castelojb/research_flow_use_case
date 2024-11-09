from research_flow.types.metrics.metric_score_model import MetricScoreModel


class MetricScore(MetricScoreModel):
    """
    Represents a metric score with a name and a value.

    Attributes:
        metric_name (str): The name of the metric.
        score (float): The value of the metric score.
    """

    metric_name: str
    score: float
