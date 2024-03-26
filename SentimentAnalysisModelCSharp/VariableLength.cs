using Microsoft.ML.Data;

namespace SentimentAnalysisModelCSharp;

internal class VariableLength
{
    [VectorType]
    public int[] VariableLengthFeatures { get; set; }
}