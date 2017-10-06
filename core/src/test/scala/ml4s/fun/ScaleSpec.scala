package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.randn
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class ScaleSpec extends Specification {
  val ε = 0.01

  "Scale" should {
    "compute scale" in {
      val factors = randn(1, 3)
      val x = randn(1, 3)
      val scale = Scale(factors, Id(3))
      val y = factors * x
      scale.maxDelta(x, y) must beCloseTo(0, ε)
    }
    "compute gradient" in {
      val factors = randn(1, 3)
      val x = randn(1, 3)
      val scale = Scale(factors, Id(3))
      scale.maxGradDelta(x, ε) must beCloseTo(0, ε)
    }
  }
}
