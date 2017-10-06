package ml4s
package fun

import org.nd4s.Implicits._

class Compose(g: Differentiable, h: Differentiable) extends Differentiable {
  require(h.out == g.in, s"Output dimension of h must mach input dimension of g ${h.out} != ${g.in}")
  val in: Int = h.in
  val out: Int = g.out

  override def apply(x: ND): ND = {
    assert(in == 0 || x.length() == in)
    g(h(x))
  }

  override def ∇(x: ND): ND = {
    val hx = h(x)
    val a = g.∇(hx)
    val b = h.∇(x)
    a ** b
  }

  override def toString(): String = s"$g($h)"
}

object Compose {
  def apply(g: Differentiable, h: Differentiable): Compose = new Compose(g, h)
}