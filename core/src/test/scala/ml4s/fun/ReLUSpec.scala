package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.randn
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class ReLUSpec extends Specification {
  lazy val ε = 0.01
  val d = 10

  "ReLu" should {
    "compute rectified linear unit" in {
      val relu = ReLU(d)
      val x = randn(d, 1)
      val y = x * x.gte(0)
      val δ = relu.maxDelta(x, y)
      δ must beCloseTo(0, ε)
    }
    "comput gradient" in {
      val relu = ReLU(d)
      val x = randn(d, 1)
      val δ = relu.maxGradDelta(x, ε)
      δ must beCloseTo(0, ε)
    }
  }
  "ReLU chained" should {
    "compute gradient" in {
      val x = randn(1, 4)
      val relu = ReLU(4)
      val δ = relu.maxGradDelta(x, ε)
      δ must beCloseTo(0, ε)
    }
  }

}
