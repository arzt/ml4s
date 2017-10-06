package ml4s.fun

object Append {
  def apply(f: Differentiable, size: Int): Differentiable = Concat(f, Id(size))
}
