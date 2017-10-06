package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{concat, zeros}
import org.nd4s.Implicits._

class MapRows(val nrows: Int, f: Differentiable) extends Differentiable {

  val in: Int = nrows * f.in

  val out: Int = nrows * f.out

  override def apply(x: ND): ND = {
    val out = (0 until in by f.in)
      .view
      .map { i =>
        x(i -> (i + f.in))
      }
      .map(f)
    concat(1, out: _*)
  }

  override def ∇(x: ND): ND = {
    val grad = zeros(out, in)
    val fout = f.out
    val fin = f.in
    (0 until nrows)
      .foreach { row =>
        val i = row * fout
        val j = row * fin
        val ri = j -> (j + fin)
        val rj = i -> (i + fout)
        val xi = x(ri)
        val g = f.∇(xi)
        grad(rj, ri) = g
      }
    grad
  }
}

object MapRows {
  def apply(rows: Int, f: Differentiable): MapRows = new MapRows(rows, f)
}