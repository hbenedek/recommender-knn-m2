package test.optimizing

import breeze.linalg._
import breeze.numerics._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import shared.predictions._
import test.shared.helpers._

class OptimizingTests extends AnyFunSuite with BeforeAndAfterAll {
  
   val separator = "\t"
   val train2Path = "data/ml-100k/u2.base"
   val test2Path = "data/ml-100k/u2.test"
   var train2 : CSCMatrix[Double] = null
   var test2 : CSCMatrix[Double] = null

   override def beforeAll{
       // For these questions, train and test are collected in a scala Array
       // to not depend on Spark
       train2 = load(train2Path, separator, 943, 1682)
       test2 = load(test2Path, separator, 943, 1682)
   }

   // Provide tests to show how to call your code to do the following tasks.
   // Ensure you use the same function calls to produce the JSON outputs in
   // the corresponding application.
   // Add assertions with the answer you expect from your code, up to the 4th
   // decimal after the (floating) point, on data/ml-100k/u2.base (as loaded above).
   test("kNN predictor with k=10") { 

    val userAverages = computeUserAverages2(train2)
    val normalizedRatings = normalizeRatings(train2, userAverages)
    val processedRatings = preProcessRatings(normalizedRatings)

    val sims = calculateCosineSimilarity(processedRatings)
    val knnSims = calculateKnnSimilarityFast(10, sims)
  
    val predictor = fitKnnPredictor(train2, 10)
    val maeKnn = evaluatePredictor(test2, predictor)

     // Similarity between user 1 and itself
     assert(within(knnSims(0,0), 0.0, 0.0001))
 
     // Similarity between user 1 and 864
     assert(within(knnSims(0, 863), 0.2423, 0.0001))

     // Similarity between user 1 and 886
     assert(within(knnSims(0,885), 0.0, 0.0001))

     // Prediction user 1 and item 1
     assert(within(predictor(0,0), 4.3191, 0.0001))

     // Prediction user 327 and item 2
     assert(within(predictor(326,1), 2.6994, 0.0001))

     // MAE on test2
     assert(within(maeKnn, 0.8287, 0.1)) 
   } 
}
