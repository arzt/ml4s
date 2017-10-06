package ml4s
package fun

object SoftmaxCols {
  def apply(rows: Int, cols: Int): Differentiable =
    Transpose(cols, rows) * SoftmaxRows(cols, rows) * Transpose(rows, cols)
}
