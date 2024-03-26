using Microsoft.ML.Data;

namespace SentimentAnalysisModelCSharp;

internal class FixedLength
{
    [VectorType(600)]
    public int[] Features { get; set; }
}