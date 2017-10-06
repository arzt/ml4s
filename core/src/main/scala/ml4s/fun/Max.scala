package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.diag
import org.nd4j.linalg.ops.transforms.Transforms.max

class Max(val in: Int, k: Double) extends Differentiable {
  val out: Int = in

  override def apply(x: ND): ND = max(x, k)

  override def ∇(x: ND): ND = diag(x gte k)
}

object Max {
  def apply(in: Int, k: Double) = new Max(in, k)

  def apply(f: Differentiable, k: Double): Differentiable = Max(f.out, k) * f
}