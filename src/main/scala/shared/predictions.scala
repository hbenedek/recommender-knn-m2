package shared

import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkContext

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
    sum(x) / x.size
  }

  def computeAllUserAverages(x: CSCMatrix[Double]): DenseVector[Double] = {
    val nonZeros = x.toDense(*,::).map(x => x.foldLeft(0)((acc, curr)=> if (curr!=0) acc + 1 else acc)).map(x => x.toDouble)
    val sums: DenseVector[Double] = sum(x.toDense(*,::)).map(x => x.toDouble)
    sums / nonZeros
  }

  def preProcessRatings(x: CSCMatrix[Double], averages: DenseVector[Double]): DenseVector[Double] ={
    for ((k,v) <- x.activeIterator){
      x(k._1, k._2) = normalizedDeviation(v, averages(k._1))
    }
    val itemNorms = sqrt(sum(dm.map(x => (x * x)).toDense(::,*))).t
    for ((k,v) <- x.activeIterator){
      x(k._1, k._2) = v / itemNorms(k._2)
    }
  }

  def calculateCosineSimilarity(x: CSCMatrix[Double]): DenseMatrix[Double] = {
      (x * x.t).toDense
  } 

  def knn(u: Int, k: Int, sims: DenseMatrix[Double]): Array[Double] = {
    val knn = sims(u,::).t.toArray.sorted(Ordering.Double.reverse).slice(1,k + 1)
  }

  def calculateKnnSimilarity(k: Int, sims: DenseMatrix[Double]) = {
    (0 to x.rows).map(u => sims(u,::).t.map(v => if (knn(u, k, sims).contains(v)) v else 0.0))
  }



}



