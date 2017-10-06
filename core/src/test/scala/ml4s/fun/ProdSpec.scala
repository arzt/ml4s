package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class ProdSpec extends Specification {
  "Prod" should {
    "compute dotwise product" in {
      val x = Nd4j.randn(1, 10)
      val a = x(0 -> 5)
      val b = x(5 ->)
      val prod = Prod(10)
      val y = a * b
      val δ = prod.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute derivative" in {
      val prod = Prod(10)
      val x = Nd4j.randn(1, 10)
      val δ = prod.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }
}
