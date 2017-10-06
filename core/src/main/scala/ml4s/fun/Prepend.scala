package ml4s.fun

object Prepend {
  def apply(size: Int, f: Differentiable): Differentiable = Concat(Id(size), f)
}
