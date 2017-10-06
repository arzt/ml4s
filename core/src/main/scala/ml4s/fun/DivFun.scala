package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{concat, diag, zeros}
import org.nd4s.Implicits._

class DivFun(val in: Int) extends Differentiable {
  val out: Int = in - 1

  override def apply(x: ND): ND = x(0 -> out) / x(out)

  override def ∇(x: ND): ND = {
    val a = zeros(out) + 1/x(out)
    val b = x(0 -> out)/x(out)/(-x(out))
    concat(1, diag(a), b.transpose())
  }
}

object DivFun {
  def apply(in: Int) = new DivFun(in)

  def apply(f: Differentiable): Differentiable = DivFun(f.out) ∘ f

  def apply(g: Differentiable, f: Differentiable): Differentiable = DivFun(g ++ f)

}