package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{concat, diag, zeros}
import org.nd4s.Implicits._

class ScaleFun(val in: Int) extends Differentiable {
  val out: Int = in - 1

  override def apply(x: ND): ND = x(0 -> out) * x(out)

  override def ∇(x: ND): ND = {
    val dia = zeros(out) + x(out)
    val hui = x(0 -> out)
    concat(1, diag(dia), hui.transpose())
  }
}

object ScaleFun {
  def apply(in: Int) = new ScaleFun(in)

  def apply(f: Differentiable): Differentiable = ScaleFun(f.out) ∘ f

  def apply(g: Differentiable, f: Differentiable): Differentiable = ScaleFun(g ++ f)

  def apply(f: Differentiable, k: Double): Differentiable = ScaleFun(f ++ Const(0, 1, k))

}