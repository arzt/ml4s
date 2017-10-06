package ml4s.fun

import org.nd4j.linalg.factory.Nd4j
import org.specs2.mutable.Specification

class ExpSpec extends Specification {
  "Exp" should {
    "compute gradient" in {
      val x = Nd4j.randn(1, 5)
      val exp = Exp(5)
      val δ = exp.maxGradDelta(x, 0.01)
      δ must beLessThanOrEqualTo(0.01)
    }
  }

}
