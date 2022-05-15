package shared

import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkContext
import ujson.Arr
import java.io.File
import breeze.util.partition


package object predictions
{
  // ------------------------ For template
  case class Rating(user: Int, item: Int, rating: Double)

  def timingInMs(f : ()=>Double ) : (Double, Double) = {
    val start = System.nanoTime() 
    val output = f()
    val end = System.nanoTime()
    return (output, (end-start)/1000000.0)
  }

  def toInt(s: String): Option[Int] = {
    try {
      Some(s.toInt)
    } catch {
      case e: Exception => None
    }
  }

  def mean(s :Seq[Double]): Double =  if (s.size > 0) s.reduce(_+_) / s.length else 0.0

  def std(s :Seq[Double]): Double = {
    if (s.size == 0) 0.0
    else { 
      val m = mean(s)
      scala.math.sqrt(s.map(x => scala.math.pow(m-x, 2)).sum / s.length.toDouble) 
    }
  }


  def load(path : String, sep : String, nbUsers : Int, nbMovies : Int) : CSCMatrix[Double] = {
    val file = Source.fromFile(path)
    val builder = new CSCMatrix.Builder[Double](rows=nbUsers, cols=nbMovies) 
    for (line <- file.getLines) {
      val cols = line.split(sep).map(_.trim)
      toInt(cols(0)) match {
        case Some(_) => builder.add(cols(0).toInt-1, cols(1).toInt-1, cols(2).toDouble)
        case None => None
      }
    }
    file.close
    builder.result()
  }

  def loadSpark(sc : org.apache.spark.SparkContext,  path : String, sep : String, nbUsers : Int, nbMovies : Int) : CSCMatrix[Double] = {
    val file = sc.textFile(path)
    val ratings = file
      .map(l => {
        val cols = l.split(sep).map(_.trim)
        toInt(cols(0)) match {
          case Some(_) => Some(((cols(0).toInt-1, cols(1).toInt-1), cols(2).toDouble))
          case None => None
        }
      })
      .filter({ case Some(_) => true
                 case None => false })
      .map({ case Some(x) => x
             case None => ((-1, -1), -1) }).collect()

    val builder = new CSCMatrix.Builder[Double](rows=nbUsers, cols=nbMovies)
    for ((k,v) <- ratings) {
      v match {
        case d: Double => {
          val u = k._1
          val i = k._2
          builder.add(u, i, d)
        }
      }
    }
    return builder.result
  }

  def partitionUsers (nbUsers : Int, nbPartitions : Int, replication : Int) : Seq[Set[Int]] = {
    val r = new scala.util.Random(1337)
    val bins : Map[Int, collection.mutable.ListBuffer[Int]] = (0 to (nbPartitions-1))
       .map(p => (p -> collection.mutable.ListBuffer[Int]())).toMap
    (0 to (nbUsers-1)).foreach(u => {
      val assignedBins = r.shuffle(0 to (nbPartitions-1)).take(replication)
      for (b <- assignedBins) {
        bins(b) += u
      }
    })
    bins.values.toSeq.map(_.toSet)
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Part BR
  /////////////////////////////////////////////////////////////////////////////////////////////////////////


  def scale(rating: Double, userAvg: Double): Double = {
    if (rating > userAvg) {5 - userAvg}
    else if (rating < userAvg) {userAvg - 1}
    else 1
  }

  def normalizedDeviation(x: Double, y: Double): Double = {
    (x - y) / scale(x,y)
  }

  def predict(userAvg: Double, itemDev: Double): Double = {
    userAvg + itemDev * scale(userAvg + itemDev, userAvg)
  }


  def globalAvg(x: CSCMatrix[Double]): Double = {
    sum(x) / x.findAll(r => r != 0.0).size
  }

  def computeUserAverages(ratings: CSCMatrix[Double]): Array[Double] = {
    val sums = Array.fill(ratings.rows)(0.0)
    val denominators = Array.fill(ratings.rows)(0.0)
    //Only iterates over non-zeros
    for ((k,v) <- ratings.activeIterator) {
      val row = k._1
      val col = k._2
      sums(row) += v
      denominators(row) += 1 
    }
    sums.zip(denominators).map{case (a, b) => a/b} 
  }
  
  def computeUserAverages2(x: CSCMatrix[Double]): DenseVector[Double] = {
    val nonZeros = x.toDense(*,::).map(x => x.foldLeft(0.0)((acc, curr)=> if (curr!=0) acc + 1 else acc)).map(x => x.toDouble)
    val sums = sum(x.toDense(*,::)).map(x => x.toDouble)
    sums / nonZeros
  }

  def normalizeRatings(x: CSCMatrix[Double], averages: DenseVector[Double]): CSCMatrix[Double] = {
    val y = new CSCMatrix.Builder[Double](rows=x.rows, cols=x.cols).result
    for ((k,v) <- x.activeIterator){
      y(k._1, k._2) = normalizedDeviation(v, averages(k._1))
    }
    y
  }

  def preProcessRatings(x: CSCMatrix[Double]): DenseMatrix[Double] ={
    val y = new CSCMatrix.Builder[Double](rows=x.rows, cols=x.cols).result
    val itemNorms = sqrt(sum(x.map(x => (x * x)).toDense, Axis._1))
    for ((k,v) <- x.activeIterator){
      y(k._1, k._2) = v / itemNorms(k._1)
    }
    y.toDense
    //normalize(x.toDense, Axis._1, 2)
    
  }

  def preProcessRatings2(x: CSCMatrix[Double]): CSCMatrix[Double] ={
    val y = new CSCMatrix.Builder[Double](rows=x.rows, cols=x.cols).result
    val itemNorms = sqrt(sum(x.map(x => (x * x)).toDense, Axis._1))
    for ((k,v) <- x.activeIterator){
      y(k._1, k._2) = v / itemNorms(k._1)
    }
    y
  }

  def calculateCosineSimilarity(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    x * x.t
  } 

  def calculateKnnSimilarityFast(k: Int, sims: DenseMatrix[Double]): DenseMatrix[Double] = {
    for (u <- 0 until sims.rows){
      val row = sims(u, ::).t.toArray.zipWithIndex
      val userKnn = row.sortWith(_._1 > _._1).slice(1, k+1).map(v => v._2)
      //val userKnn = argtopk(sims(u, ::).t,k+1).toArray.slice(1, k + 1)
      for (v <- 0 until sims.cols) {
        if (!userKnn.contains(v)) sims(u, v) = 0.0
      }
    }
    sims
  }

  def calculateItemDevs(xNormalized: CSCMatrix[Double], x: CSCMatrix[Double], sims: DenseMatrix[Double]): DenseMatrix[Double] = {
    val indicator = x.toDense.map(v => if (v != 0.0) 1.0 else 0.0)
    val itemDevs = (sims * xNormalized.toDense) /:/ (abs(sims) * indicator)
    itemDevs.map(v => if (v.isNaN()) 0.0 else v)
    }

  def fitKnnPredictor(x: CSCMatrix[Double], k: Int) : (Int, Int) => Double = {
    val userAvgs = computeUserAverages2(x)
    val normalizedRatings = normalizeRatings(x, userAvgs)
    val preprocessedRatings = preProcessRatings(normalizedRatings)
    val simsCos = calculateCosineSimilarity(preprocessedRatings)
    val simsKnn = calculateKnnSimilarityFast(k, simsCos.copy)
    val itemDevs = calculateItemDevs(normalizedRatings, x, simsKnn)    

    (u: Int, i: Int) => predict(userAvgs(u), itemDevs(u,i))
  }

  def evaluatePredictor(test: CSCMatrix[Double], predictor: (Int, Int) => Double): Double = {
    val errors = (for ((k,v) <- test.activeIterator) yield (predictor(k._1, k._2) - v).abs).toList
    errors.reduce(_ + _) / errors.size
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Part EK
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  def calculateParallelKnn(xPreprocessed: CSCMatrix[Double], sc: SparkContext, k: Int): CSCMatrix[Double] = {
    val nbUsers = xPreprocessed.rows
    val broadcast = sc.broadcast(xPreprocessed.toDense)
    val topks = sc.parallelize(0 to nbUsers - 1).mapPartitions(iter => for {u <- iter;
      val r = broadcast.value
      val sims = r * r.t(::,u)
      val knn = sims.toArray.zipWithIndex.sortWith(_._1 > _._1).slice(1, k+1).map(v => (u, v._2, sims(v._2)))
    } yield knn).collect().flatMap(x => x)
    
    val builder = new CSCMatrix.Builder[Double](rows=nbUsers, cols=nbUsers)
    for ((u,v,s) <- topks) {builder.add(u, v, s)}
    builder.result
  }

  def calculateParallelItemDevs(xNormalized: CSCMatrix[Double], x: CSCMatrix[Double], sims: CSCMatrix[Double], sc: SparkContext): CSCMatrix[Double] = {
    val xBroadcast = sc.broadcast(x.toDense)
    val xNormalizedBroadcast = sc.broadcast(xNormalized.toDense)
    val nbItems = x.cols
    val nbUsers = x.rows
    val allDevs = sc.parallelize(0 to nbItems - 1).mapPartitions(iter => for {i <- iter;
      val iIndicator = xBroadcast.value(::,i).map(v => if (v != 0.0) 1.0 else 0.0)
      val iNormalized = xNormalizedBroadcast.value(::,i)
      val iDevs = ((sims * iNormalized) /:/ (abs(sims) * iIndicator)).toArray.zipWithIndex.map{case (d,u) => (u,i,d)}
    } yield iDevs).collect().flatMap(x => x)

    val builder = new CSCMatrix.Builder[Double](rows=nbUsers, cols=nbItems)
    for ((u,i,d) <- allDevs) {builder.add(u, i, d)}
    builder.result
    }

   def fitParallelKnn(x: CSCMatrix[Double], sc: SparkContext, k: Int): (Int, Int) => Double ={
    val userAvgs = computeUserAverages2(x)
    val normalizedRatings = normalizeRatings(x, userAvgs)
    val preprocessedRatings = preProcessRatings2(normalizedRatings)

    val knnSims = calculateParallelKnn(preprocessedRatings, sc, k)
    //val itemDevs = calculateParallelItemDevs(normalizedRatings, x, knnSims, sc)
    val itemDevs = calculateItemDevs(normalizedRatings, x, knnSims.toDense)

    (u: Int, i: Int) => predict(userAvgs(u), itemDevs(u,i))
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Part AK
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  def fitApproximateKnn(x: CSCMatrix[Double], k: Int, replication: Int, partition: Int, sc: SparkContext): (Int, Int) => Double ={
    val userAvgs = computeUserAverages2(x)
    val normalizedRatings = normalizeRatings(x, userAvgs)
    val preprocessedRatings = preProcessRatings2(normalizedRatings)
    val approxKnnSims = calculateApproximateKnn(preprocessedRatings, k, replication, partition, sc)
    val itemDevs = calculateItemDevs(normalizedRatings, x, approxKnnSims.toDense)
    (u: Int, i: Int) => predict(userAvgs(u), itemDevs(u,i))
  }

  def calculateApproximateKnn(preprocessed: CSCMatrix[Double], k: Int, replication: Int, nbPartitions: Int, sc: SparkContext): CSCMatrix[Double] = {
     val nbUsers = preprocessed.rows
     val partitioned = partitionUsers(nbUsers, nbPartitions, replication)
     val br = sc.broadcast(preprocessed.toDense)
     val knns = sc.parallelize(partitioned).map(p => {
       val r = br.value
       val slice = r(p.toSeq.sortWith(_ < _), ::).toDenseMatrix
       val cosSims = calculateCosineSimilarity(slice)
       val knn = (0 until cosSims.rows).toList.map(u => cosSims(u, ::).t
                                                                .toArray
                                                                .zip(p.toSeq.sortWith(_ < _))
                                                                .sortWith(_._1 > _._1)
                                                                .slice(1, k+1)
                                                                .map(v => (u, v._2, v._1))).flatMap(x => x)
       val userMap = (0 until cosSims.rows).zip(p.toSeq.sortWith(_ < _)).toMap
       knn.map{case (u, v, s) => (userMap(u), v, s)}
     }).collect()
   
    val flattened = knns.flatMap(x => x)
                        .groupBy(x => x._1)
                        .map(x => x._2.map(y => (y._2, y._3))
                                                .toList
                                                .sortWith(_._2 > _._2)
                                                .slice(0, k)
                                                .map(z => (x._1, z._1, z._2)))
                        .flatMap(x => x)
  
    // in the end we build the sparse matrix out of the tuples similarly as before
    val builder = new CSCMatrix.Builder[Double](rows=nbUsers, cols=nbUsers)
    for ((u,v,s) <- flattened) {builder.add(u, v, s)}
    builder.result
  }

  def evaluateReplications(replications: List[Int], x: CSCMatrix[Double], test: CSCMatrix[Double], k: Int, partitions: Int, sc: SparkContext):  Map[Int,Double] ={ 
     replications.map(r => {
       val predictor = fitApproximateKnn(x, k, r, partitions, sc)
       (r, evaluatePredictor(test, predictor))
      }).toMap
    }

}




