package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{zeros, create}
import org.nd4s.Implicits._

class Const(val in: Int, value: ND) extends Differentiable {
  val out: Int = value.length()

  def apply(x: ND): ND = {
    assert(in == 0 || x.length() == in)
    value.ravel()
  }

  override def ∇(x: ND): ND = if (in == 0) null else zeros(out, in)

  override def toString(): String = "Const"
}

object Const {
  def apply(in: Int, value: ND): Const = new Const(in, value)

  def apply(in: Int, out: Int, value: Double): Const = Const(in, create(out) + value)
}