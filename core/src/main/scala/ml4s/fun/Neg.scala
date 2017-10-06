package ml4s
package fun

object Neg {
  def apply(f: Differentiable): Differentiable = ScaleFun(f, -1)

  def apply(dim: Int): Differentiable = Neg(Id(dim))
}