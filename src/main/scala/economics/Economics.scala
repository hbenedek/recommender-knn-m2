import org.rogach.scallop._
import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import ujson._
import scala.math._

package economics {

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val json = opt[String]()
  verify()
}

object Economics {
  def main(args: Array[String]) {
    println("")
    println("******************************************************")

    var conf = new Conf(args)


    //E1

    def minRentingDays(buyingCost: Double, rentingCost: Double): Double = {
     (buyingCost / rentingCost).ceil
    }
    val e1 = minRentingDays(38600.0, 20.40)

    //E2

    def containerDailyCost(vCPUCostPerSec: Double , nbvCPU: Int, gbCostPerSec: Double, nbGB: Int): Double = {
      (nbvCPU * vCPUCostPerSec + nbGB * gbCostPerSec) * 60 * 60 * 24 
    }
    //Since 4 RPis have 4*8GB of RAM and 1 vCPU is roughly equivalent to the performance of 4 PRis
    val e21 = containerDailyCost(1.14e-6, 1, 1.6e-7, 32)

    def rPisDailyCost(nbRpis: Int, power: Double, energyCost: Double): Double = {
      nbRpis * power * energyCost * 24 / 1000
    }

    val e22 = rPisDailyCost(4, 3, 0.25)
    val e23 = rPisDailyCost(4, 4, 0.25)

    def minRentingDaysRPiPower(buyingCost: Double, rentingCost: Double): Double = {
      (buyingCost / rentingCost).ceil
    }

    val e24 = minRentingDaysRPiPower(108.48, e22)
    val e25 = minRentingDaysRPiPower(108.48, e23)

    //E3

    def nbRPisEqBuyingICCM7(iccm7BuyingCost: Double, rpisBuyingCost: Double): Int = {
      (iccm7BuyingCost / rpisBuyingCost).floor.toInt
    }

    val e31 = nbRPisEqBuyingICCM7(38600 ,108.48)

    def ratioRAM(nbRPis: Int, ramRPis: Int, ramICC: Int): Double = {
      ramICC.toDouble / (nbRPis * ramRPis).toDouble
    }

    val e32 = ratioRAM(e31, 8, 24 * 64)

    def ratioCompute(nbCoresICC: Int, throughPutICC: Double, nbRPis: Int, nbCoresRPis: Int, throughPutRPis: Double): Double = {
      (nbCoresICC *  throughPutICC) / (nbRPis * nbCoresRPis * throughPutRPis)
    } 

    val e33 = ratioCompute(2 * 14, 2.6, e31, 4, 1.5)

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
          "E.1" -> ujson.Obj(
            "MinRentingDays" -> ujson.Num(e1) // Datatype of answer: Double
          ),
          "E.2" -> ujson.Obj(
            "ContainerDailyCost" -> ujson.Num(e21),
            "4RPisDailyCostIdle" -> ujson.Num(e22),
            "4RPisDailyCostComputing" -> ujson.Num(e23),
            "MinRentingDaysIdleRPiPower" -> ujson.Num(e24),
            "MinRentingDaysComputingRPiPower" -> ujson.Num(e25) 
          ),
          "E.3" -> ujson.Obj(
            "NbRPisEqBuyingICCM7" -> ujson.Num(e31),
            "RatioRAMRPisVsICCM7" -> ujson.Num(e32),
            "RatioComputeRPisVsICCM7" -> ujson.Num(e33)
          )
        )

        val json = write(answers, 4)
        println(json)
        println("Saving answers in: " + jsonFile)
        printToFile(json, jsonFile)
      }
    }

    println("")
  } 
}

}
