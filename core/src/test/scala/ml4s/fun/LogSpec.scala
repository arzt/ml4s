package ml4s.fun

import org.nd4j.linalg.factory.Nd4j.rand
import org.nd4s.Implicits._
import org.specs2.mutable.Specification


class LogSpec extends Specification {
  "Log" should {
    "compute gradient" in {
      val x = rand(1, 5) + 1
      val log = Log(5)
      val δ = log.maxGradDelta(x, 0.01)
      δ must beLessThanOrEqualTo(0.01)
    }
  }

}
