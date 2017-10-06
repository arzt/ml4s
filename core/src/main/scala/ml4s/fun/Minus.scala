package ml4s
package fun

object Minus {
  /*
  * */
  def apply(a: Differentiable, b: Differentiable): Differentiable = Sum(a, Scale(-1, b))

  def apply(a: Differentiable, b: ND): Differentiable = Sum(a, Scale(-1, Const(0, b)))

  def apply(a: ND, b: Differentiable): Differentiable = Sum(Const(0, a), Scale(-1, b))

  def apply(a: ND, b: ND): Differentiable = Sum(Const(a.rows(), a), Scale(-1, Const(b.rows(), b)))

}
