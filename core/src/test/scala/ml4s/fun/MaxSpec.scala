package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.create
import org.nd4j.linalg.ops.transforms.Transforms
import org.specs2.mutable.Specification

class MaxSpec extends Specification {
  "Max" should {
    "compute pairwise max" in {
      val k = 0
      val f = Max(4, k)
      val x = create(Array[Double](-1, 2, -3, 9), Array(1, 4))
      val y = Transforms.max(x, k)
      val δ = f.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute derivativen" in {
      val f = Max(4, 0)
      val data = Array[Double](-1, 2, -3, 9)
      val x = create(data, Array(1, 4))
      val δ = f.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }

}
