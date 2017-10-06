package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.randn
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class SoftmaxColsSpec extends Specification {
  "SoftMaxCols" should {
    "comput column-wize softmax" in {
      val f = SoftmaxCols(10, 5)
      val data = randn(10, 5)
      val y = f(data.ravel()).reshape(10, 5)
      val y2 = softmax(data.transpose()).transpose()
      val δ = (y - y2).maxNumber().doubleValue()
      δ must beCloseTo(0, 0.01)
    }
  }
}
