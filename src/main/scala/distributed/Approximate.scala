import org.rogach.scallop._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import ujson._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import shared.predictions._

package distributed {

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val k = opt[Int]()
  val json = opt[String]()
  val users = opt[Int]()
  val movies = opt[Int]()
  val separator = opt[String](default=Some("\t"))
  val replication = opt[Int](default=Some(1))
  val partitions = opt[Int](default=Some(1))
  val master = opt[String]()
  val num_measurements = opt[Int](default=Some(1))
  verify()
}

object Approximate {
  def main(args: Array[String]) {
    var conf = new Conf(args)

    // Remove these lines if encountering/debugging Spark
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val spark = conf.master.toOption match {
      case None => SparkSession.builder().getOrCreate();
      case Some(master) => SparkSession.builder().master(master).getOrCreate();
    }
    val sc = spark.sparkContext

    println("")
    println("******************************************************")

    // conf object is not serializable, extract values that
    // will be serialized with the parallelize implementations
    val conf_users = conf.users()
    val conf_movies = conf.movies()
    val conf_k = conf.k()

    println("Loading training data")
    val train = loadSpark(sc, conf.train(), conf.separator(), conf.users(), conf.movies())
    val test = loadSpark(sc, conf.test(), conf.separator(), conf.users(), conf.movies())
    var knn : CSCMatrix[Double] = null

    println("Partitioning users")
    var partitionedUsers : Seq[Set[Int]] = partitionUsers(
      conf.users(), 
      conf.partitions(), 
      conf.replication()
    )
    val measurements = (1 to scala.math.max(1,conf.num_measurements()))
      .map(_ => timingInMs( () => {
      val predictor = fitApproximateKnn(train, conf.k(), conf.replication(), conf.partitions(), sc)
      val mae = evaluatePredictor(test, predictor)
      mae
    }))
    val mae = measurements(0)._1
    val timings = measurements.map(_._2)


    //AK.1
    val userAvgs = computeUserAverages2(train)
    val normalizedRatings = normalizeRatings(train, userAvgs)
    val preprocessedRatings = preProcessRatings2(normalizedRatings)

    val approxSims = calculateApproximateKnn(preprocessedRatings, conf.k(), conf.replication(), conf.partitions(), sc)


    //AK.2
    val replications = List(1,2,3,4,6,8)
    val maes = evaluateReplications(replications, train, test, 300, conf.partitions(), sc).toList.map{case (k,m)=>List(k,m)}

    // Save answers as JSON
    def printToFile(content: String,
                    location: String = "./answers.json") =
      Some(new java.io.PrintWriter(location)).foreach{
        f => try{
          f.write(content)
        } finally{ f.close }
    }
    conf.json.toOption match {
      case None => ;
      case Some(jsonFile) => {
        val answers = ujson.Obj(
          "Meta" -> ujson.Obj(
            "train" -> ujson.Str(conf.train()),
            "test" -> ujson.Str(conf.test()),
            "k" -> ujson.Num(conf.k()),
            "users" -> ujson.Num(conf.users()),
            "movies" -> ujson.Num(conf.movies()),
            "master" -> ujson.Str(sc.getConf.get("spark.master")),
            "num-executors" -> ujson.Str(if (sc.getConf.contains("spark.executor.instances"))
                                            sc.getConf.get("spark.executor.instances")
                                         else
                                            ""),
            "num_measurements" -> ujson.Num(conf.num_measurements()),
            "partitions" -> ujson.Num(conf.partitions()),
            "replication" -> ujson.Num(conf.replication()) 
          ),
          "AK.1" -> ujson.Obj(
            "knn_u1v1" -> ujson.Num(approxSims(0,0)),
            "knn_u1v864" -> ujson.Num(approxSims(0,863)),
            "knn_u1v344" -> ujson.Num(approxSims(0,343)),
            "knn_u1v16" -> ujson.Num(approxSims(0,15)),
            "knn_u1v334" -> ujson.Num(approxSims(0,333)),
            "knn_u1v2" -> ujson.Num(approxSims(0,1))
          ),
          "AK.2" -> ujson.Obj(
            "mae" -> maes
          ),
          "AK.3" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings)),
            "stddev (ms)" -> ujson.Num(std(timings))
          )
        )
        val json = write(answers, 4)

        println(json)
        println("Saving answers in: " + jsonFile)
        printToFile(json, jsonFile)
      }
    }

    println("")
    spark.stop()
  } 
}

}
