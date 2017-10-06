package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.specs2.mutable.Specification

class SigmoidSpec extends Specification {
  "Sigmoid" should {
    "compute sigmoid" in {
      val x = randn(1, 5)
      val y = sigmoid(x)
      val sig = Sigmoid(5)
      val δ = sig.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute gradient" in {
      val x = randn(1, 5)
      val log = Sigmoid(5)
      val δ = log.maxGradDelta(x, 0.01)
      δ must beLessThanOrEqualTo(0.01)
      1 === 1
    }
  }

}
