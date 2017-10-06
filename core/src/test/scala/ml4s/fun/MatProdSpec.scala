package ml4s.fun

import org.nd4j.linalg.factory.Nd4j.{concat, randn}
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class MatProdSpec extends Specification {


  def measure(x: => Unit, times: Int = 10): Double = {
    import System.{currentTimeMillis => now}
    var i = 0
    val ti = new Array[Double](times)
    while (i < times) {
      val a = now
      x
      val b = now
      ti(i) = b - a
      i = i + 1
    }
    ti.sum / times
  }

  val ε = 0.01
  val n = 4
  val m = 5
  val l = 3

  "MatProd" should {
    "compute matrix product" in {
      val dot = MatProd(m, l, n)
      val x1 = randn(m, l)
      val x2 = randn(l, n)
      val x = concat(0, x1.ravel().transpose(), x2.ravel().transpose())
      val y = (x1 ** x2).ravel().transpose()
      val δ = dot.maxDelta(x, y)
      δ must beCloseTo(0, ε)
    }
    "compute gradient" in {
      val dot = MatProd(m, l, n)
      val x1 = randn(m, l)
      val x2 = randn(l, n)
      val x = concat(0, x1.ravel().transpose(), x2.ravel().transpose())
      val δ = dot.maxGradDelta(x, ε)
      δ must beCloseTo(0, ε)
    }
    "take time in" in {
      val a = 20
      val b = 200
      val c = 20
      val dot = MatProd(a, b, c)
      val x1 = randn(a, b)
      val x2 = randn(b, c)
      val x = concat(0, x1.ravel().transpose(), x2.ravel().transpose())
      val tb = measure(dot.old(x), 100)
      val ta = measure(dot.∇(x), 100)
      println(ta)
      println(tb)
      1 === 1
    }
  }
}
