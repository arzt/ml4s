package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{diag, zeros}
import org.nd4s.Implicits._

class Scale(scale: ND) extends Differentiable {
  assert(scale.isVector)
  val in: Int = scale.length()
  val out: Int = scale.length()

  override def apply(x: ND): ND = {
    assert(x.length() == in)
    x * scale
  }

  override def ∇(x: ND): ND = diag(scale)
}

object Scale {

  def apply(scale: ND): Scale = new Scale(scale)

  def apply(scale: Double, dim: Int): Scale = Scale(zeros(1, dim) + scale)

  def apply(scale: ND, f: Differentiable): Differentiable = Scale(scale) ∘ f

  def apply(scale: Double, f: Differentiable): Differentiable = Scale(scale, f.out) ∘ f

}