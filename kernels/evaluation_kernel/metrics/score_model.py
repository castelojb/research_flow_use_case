from research_flow.types.metrics.metric_score_model import MetricScoreModel


class MetricScore(MetricScoreModel):
    metric_name: str
    score: float
