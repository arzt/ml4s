package ml4s.fun

import org.nd4j.linalg.factory.Nd4j.randn
import org.specs2.mutable.Specification

class IdSpec extends Specification {
  val ε = 0.01

  "Identity" should {
    val id = Id(3)
    val x = randn(1, 3)
    "compute identity" in {
      id.maxDelta(x, x) must beCloseTo(0, ε)
    }
    "compute gradient" in {
      id.maxGradDelta(x, ε) must beCloseTo(0, ε)
    }
  }
}
