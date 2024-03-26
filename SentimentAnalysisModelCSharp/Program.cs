// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using SentimentAnalysisModelCSharp;
using System;
using System.IO;
using System.Collections.Generic;

const int FeatureLength = 600;
string _modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model");

var mlContext = new MLContext();
var lookupMap = mlContext.Data.LoadFromTextFile(Path.Combine(_modelPath, "imdb_word_index.csv"), columns: new[]
    {
        new TextLoader.Column("Words", DataKind.String, 0),
        new TextLoader.Column("Ids", DataKind.Int32, 1)
    },
    separatorChar: ','
);

Action<VariableLength, FixedLength> ResizeFeaturesAction = (s, f) =>
{
    var features = s.VariableLengthFeatures;
    Array.Resize(ref features, FeatureLength);
    f.Features = features;
};


var tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);

DataViewSchema schema = tensorFlowModel.GetModelSchema();

Console.WriteLine("============ TensorFlow Model Schema =============");
var featuresType = (VectorDataViewType)schema["Features"].Type;

Console.WriteLine($"Name: Features, Type: {featuresType.ItemType.RawType}, Size: {featuresType.Dimensions[0]}");

var predictionType = (VectorDataViewType)schema["Prediction/Softmax"].Type;

Console.WriteLine($"Name: Prediction/Softmax, Type: {predictionType.ItemType.RawType}, Size: ({predictionType.Dimensions[0]})");
Console.WriteLine("Press Enter to continue...");
Console.ReadLine();

IEstimator<ITransformer> pipeline = mlContext.Transforms.Text.TokenizeIntoWords("TokenizedWords", "ReviewText")
    .Append(mlContext.Transforms.Conversion.MapValue("VariableLengthFeatures", lookupMap, lookupMap.Schema["Words"], lookupMap.Schema["Ids"], "TokenizedWords"))
    .Append(mlContext.Transforms.CustomMapping(ResizeFeaturesAction,"Resize"))
    .Append(tensorFlowModel.ScoreTensorFlowModel("Prediction/Softmax", "Features"))
    .Append(mlContext.Transforms.CopyColumns("Prediction", "Prediction/Softmax"));

IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<MovieReview>());
ITransformer model = pipeline.Fit(dataView);

PredictionSentiment(mlContext, model);

static void PredictionSentiment(MLContext mlContext, ITransformer model)
{
    var engine = mlContext.Model.CreatePredictionEngine<MovieReview, SentimentPrediction>(model);

    var review = new MovieReview()
    {
        ReviewText = "This is a interesting movie1"
    };

    var sentimentPrediction = engine.Predict(review);

    Console.WriteLine($"Number of classes: {sentimentPrediction.Prediction.Length}");
    Console.WriteLine($"Is sentiment/review positive? {(sentimentPrediction.Prediction[1] > 0.5 ? "Yes" : "No")}");
    Console.WriteLine("Press Enter to continue...");
    Console.ReadLine();
}