package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class SquaredDiffSpec extends Specification {
  "Squared diff" should {
    "comute" in {
      val label = Nd4j.randn(1, 4)
      val label2 = Nd4j.randn(1, 4)
      val diff = SquaredDiff(Id(4), Const(0, label2))
      val a = label - label2
      val y = a * a
      val δ = diff.maxDelta(label, y)
      δ must beLessThan(0.01)
    }
  }
}
