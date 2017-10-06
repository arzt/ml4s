package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.diag
import org.nd4s.Implicits._

class Square(val in: Int) extends Differentiable {
  lazy val out: Int = in

  override def ∇(x: ND): ND = diag(x * 2)

  override def apply(x: ND): ND = {
    assert(x.length() == in, s"Dimensions disagree ${x.length()} != $in")
    x * x
  }
}

object Square {
  def apply(in: Int): Square = new Square(in)

  def apply(fun: Differentiable): Compose = Compose(Square(fun.out), fun)
}