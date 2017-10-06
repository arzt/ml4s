package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms
import org.specs2.mutable.Specification
import org.nd4s.Implicits._

class MeanSquaredErrorSpec extends Specification {
  "Mean squared error" should {
    "compute error" in {
      val label = randn(1, 4)
      val y = zeros(1, 1)
      val mse = MeanSquaredError(Id(4), Const(0, label))
      val δ = mse.maxDelta(label, y)
      δ must beLessThan(0.01)
    }
    "compute error2" in {
      val a = randn(1, 4)
      val b = randn(1, 4)
      val y = ((a - b)*(a - b)).sum(0,1)/2.0/4
      val mse = MeanSquaredError(Id(4), Const(0, b))
      val δ = mse.maxDelta(a, y)
      δ must beLessThan(0.01)
    }
  }
}
