package ml4s.fun

object SoftmaxRows {
  def apply(rows: Int, cols: Int): Differentiable = MapRows(rows, Softmax(cols))
}
