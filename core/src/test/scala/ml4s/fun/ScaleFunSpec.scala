package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class ScaleFunSpec extends Specification {
  "Scale fun" should {
    "scale a function" in {
      val x = Nd4j.randn(1, 4)
      val y = x(0 -> 3) * x(3)
      val f = ScaleFun(4)
      val δ = f.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute derivative" in {
      val x = Nd4j.randn(1, 4)
      val f = ScaleFun(4)
      val δ = f.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }
}
