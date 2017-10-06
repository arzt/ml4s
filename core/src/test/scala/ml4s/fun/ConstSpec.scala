package ml4s.fun

import org.nd4j.linalg.factory.Nd4j.randn
import org.specs2.mutable.Specification

class ConstSpec extends Specification {

  val ε = 0.01

  "Const" should {
    "compute a constant" in {
      val y = randn(1, 3)
      val x = randn(1, 5)
      val const = Const(5, y)
      const.maxDelta(x, y) must beCloseTo(0, ε)
    }
    "compute gradient 2" in {
      val y = randn(1, 3)
      val x = randn(1, 5)
      val const = Const(5, y)
      const.maxGradDelta(x, ε) must beCloseTo(0, ε)
    }
  }
}
