package ml4s
package fun

import org.nd4s.Implicits._
import org.nd4j.linalg.factory.Nd4j.diag
import org.nd4j.linalg.ops.transforms.Transforms.abs

class ReLU(val in: Int) extends Differentiable {
  val out: Int = in

  override def ∇(x: ND): ND = diag(x gte 0)

  override def apply(x: ND): ND = abs(x) * (x gte 0)
}

object ReLU {
  def apply(in: Int): Differentiable = new ReLU(in)

  def apply(in: Differentiable): Differentiable = Compose(ReLU(in.out), in)
}