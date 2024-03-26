using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace MachineLearningModelCSharp
{
    internal class ClusterPrediction
    {
        [ColumnName("PredictedLabel")] public uint PredictedClusterId;
        [ColumnName("Score")] public float[] Distances;
    }
}
