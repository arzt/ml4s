package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.specs2.mutable.Specification

class SoftmaxSpec extends Specification {
  "Soft max" should {
    "compute soft max function" in {
      val in = 10
      val x = Nd4j.randn(1, in)
      val softMax = Softmax(in)
      val y = Transforms.softmax(x)
      val δ = softMax.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute derivative" in {
      val in = 10
      val x = Nd4j.randn(1, in)
      val softMax = Softmax(in)
      val δ = softMax.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }

}
