package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{zeros, diag}
import org.nd4s.Implicits._

class Reciprocal(val in: Int) extends Differentiable {
  val out: Int = in

  override def apply(x: ND): ND = {
    require(x.length() == in)
    (0 until in)
      .foreach { i =>
        x(i) = 1.0 / x(i)
      }
    x
  }

  override def ∇(x: ND): ND = {
    val y = zeros(1, in)
    (0 until in)
      .foreach { i =>
        y(i) = -1.0 / x(i) / x(i)
      }
    diag(y)
  }
}

object Reciprocal {
  def apply(in: Int) = new Reciprocal(in)

  def apply(f: Differentiable): Differentiable = Reciprocal(f.out) ∘ f
}