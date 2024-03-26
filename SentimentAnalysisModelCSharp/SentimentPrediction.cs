using Microsoft.ML.Data;

namespace SentimentAnalysisModelCSharp;

internal class SentimentPrediction
{
    [VectorType(2)]
    public float[] Prediction { get; set; }
}