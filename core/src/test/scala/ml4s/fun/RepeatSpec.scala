package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{concat, randn}
import org.specs2.mutable.Specification

class RepeatSpec extends Specification {
  "Repeat" should {
    "compute fan function" in {
      val x = randn(1, 5)
      val fan = Repeat(5, 3)
      val y = concat(1, x, x, x)
      val δ = fan.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "comput derivative" in {
      val x = randn(1, 5)
      val fan = Repeat(5, 3)
      val δ = fan.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }
}
