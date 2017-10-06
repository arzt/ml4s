package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.rand
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class DivFunSpec extends Specification {
  "Div fun" should {
    "divide a function" in {
      val x = rand(1, 4)
      x(3) = x(3) + 1
      val y = x(0 -> 3) / x(3)
      val f = DivFun(4)
      val δ = f.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute derivative" in {
      val x = rand(1, 4)
      x(3) = x(3) + 1
      val f = DivFun(4)
      val δ = f.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }
}
