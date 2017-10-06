package ml4s

package fun

import org.nd4j.linalg.ops.transforms.Transforms.exp
import org.nd4j.linalg.factory.Nd4j.diag


class Exp(val in: Int) extends Differentiable {
  def out: Int = in

  override def apply(x: ND): ND = exp(x)

  override def ∇(x: ND): ND = diag(exp(x))

  override def toString(): String = s"Exp-$in-$out"
}

object Exp {
  def apply(in: Int): Exp = new Exp(in)

  def apply(f: Differentiable): Differentiable = Exp(f.out) ∘ f
}