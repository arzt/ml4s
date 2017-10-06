package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j._
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class SumSpec extends Specification {
  "Sum" should {
    "compute sum" in {
      val x = randn(5, 10)
      val f = Sum(10, 5)
      val y = x.sum(0).ravel()
      val δ = f.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute sum of multiple functions" in {
      val fa = Id(5)
      val x = rand(1, 5) + 1
      val ya = fa(x)

      val fb = Exp(5)
      val yb = fb(x)

      val fc = Log(5)
      val yc = fc(x)

      val sum = Sum(fa, fb, fc)
      val y = ya + yb + yc
      val δ = sum.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute derivative" in {
      val x = randn(5, 10).ravel()
      val f = Sum(10, 5)
      val δ = f.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }
}
