package com.berkgokden

// $example on$
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
// $example off$
import org.apache.spark.sql.SparkSession

/**
  * An example demonstrating ALS.
  * Run with
  * {{{
  * bin/run-example ml.ALSExample
  * }}}
  */
object Sparktut extends App{

  // $example on$
  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }
  // $example off$

  val spark = SparkSession
    .builder
    .master("local")
    .appName("ALSExample")
    .getOrCreate()
  import spark.implicits._

  val path = getClass.getResource("/sample_movielens_ratings.txt").getPath
  println(path)
  // $example on$
  val ratings = spark.read.textFile(path)
    .map(parseRating)
    .toDF()
  val Array(training, test) = ratings.randomSplit(Array(0.80, 0.20))

  // Build the recommendation model using ALS on the training data
  val als = new ALS()
    .setMaxIter(5)
    .setRegParam(0.01)
    .setUserCol("userId")
    .setItemCol("movieId")
    .setRatingCol("rating")
  val model = als.fit(training)

  // Evaluate the model by computing the RMSE on the test data
  val predictions = model.transform(test)


  val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setLabelCol("rating")
    .setPredictionCol("prediction")
  val rmse = evaluator.evaluate(predictions)
  println(s"Root-mean-square error = $rmse")

  // $example off$
  // predictions.take(3).foreach(c => println(c.getAs[Int]("userId") + ":" + c.getAs[Float]("rating") + ":" + c.getAs[Float]("prediction")))
  val result = predictions.filter(c => c.getAs[Int]("userId") == 1).sort($"prediction".desc)
    .map(c => c.getAs[Int]("userId") + ":" + c.getAs[Int]("movieId") + ":"+ c.getAs[Float]("rating") + ":" + c.getAs[Float]("prediction"))
    .collect()


  spark.stop()

  result.foreach(s => println(s))
  println(s"Root-mean-square error = $rmse")

}
// scalastyle:on println
