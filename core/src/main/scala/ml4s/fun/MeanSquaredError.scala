package ml4s
package fun

object MeanSquaredError {
  def apply(a: Differentiable, b: Differentiable): Differentiable = {
    val N = b.out
    //Scale(1.0/N, Sum(HalfSquare(Minus(a, b))))
    ScaleFun(Sum(HalfSquare(Minus(a, b))), 1.0/N)
  }
}
