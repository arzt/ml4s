package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{diag, ones}

class Id(val in: Int) extends Differentiable {

  val out: Int = in

  override def apply(x: ND): ND = {
    assert(x.length() == in)
    x
  }

  override def ∇(x: ND): ND = diag(ones(in))

  override def toString: String = s"Id($in)"
}

object Id {
  def apply(in: Int): Id = new Id(in)
}