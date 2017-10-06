package ml4s
package fun

object SquaredDiff {
  def apply(a: Differentiable, b: Differentiable): Compose = Square(Minus(a, b))
}
