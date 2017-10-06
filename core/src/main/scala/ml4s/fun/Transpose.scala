package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{concat, zeros}
import org.nd4s.Implicits._


class Transpose(rows: Int, cols: Int) extends Differentiable {

  val in: Int = rows * cols
  val out: Int = in

  override def apply(x: ND): ND = x.reshape(rows, cols).transpose().ravel()

  override def ∇(x: ND): ND = {
    val b = 0.until(cols).view
      .map { col =>
        val a = 0.until(rows).view
          .map { row =>
            zeros(rows, cols)(row, col) = 1
          }
        concat(1, a: _*)
      }
    concat(0, b: _*)
  }
}

object Transpose {
  def apply(rows: Int, cols: Int): Transpose = new Transpose(rows, cols)

  def apply(rows: Int, f: Differentiable): Differentiable = new Transpose(rows, f.out / rows) ∘ f
}