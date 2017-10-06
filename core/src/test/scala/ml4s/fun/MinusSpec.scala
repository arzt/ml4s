package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j._
import org.specs2.mutable.Specification
import org.nd4s.Implicits._

class MinusSpec extends Specification {
  "Minus" should {
    "compute minus" in {
      val label = randn(1,4)
      val minus = Minus(Id(4), label)
      val y = label - label
      val δ = minus.maxDelta(label, y)
      δ must beLessThan(0.01)
    }
  }

}
