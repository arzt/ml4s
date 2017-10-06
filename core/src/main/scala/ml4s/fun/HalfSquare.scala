package ml4s.fun

import ml4s.ND
import org.nd4j.linalg.factory.Nd4j.diag
import org.nd4s.Implicits._

class HalfSquare(val in: Int) extends Differentiable {
  val out: Int = in

  override def apply(x: ND): ND = x * x / 2

  override def ∇(x: ND): ND = diag(x)
}

object HalfSquare {
  def apply(in: Int): HalfSquare = new HalfSquare(in)

  def apply(in: Differentiable): Differentiable = HalfSquare(in.out) ∘ in
}
