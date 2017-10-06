package ml4s.fun

object Mean {
  def apply(in: Int): Differentiable = Scale(1.0/in, Sum(in))
  def apply(in: Differentiable): Differentiable = Scale(1.0/in.out, Sum(in))
}
