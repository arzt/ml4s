package ml4s.fun

import ml4s.ND
import org.nd4j.linalg.factory.Nd4j.{concat, zeros}
import org.nd4s.Implicits._

class Concat(g: Differentiable, h: Differentiable) extends Differentiable {

  val in: Int = g.in + h.in

  val out: Int = g.out + h.out

  override def apply(x: ND): ND = {
    assert(in == 0 || x.length() == in)
    val xg = if (g.in > 0) x(0 -> g.in) else null
    val xh = if (h.in > 0) x(g.in ->) else null
    val out = zeros(1, g.out + h.out)
    out(0 -> g.out) = g(xg)
    out(g.out ->) = h(xh)
    out
  }

  override def ∇(x: ND): ND = {
    (g.in, h.in) match {
      case (0, 0) =>
        null
      case (gin, 0) =>
        val out = zeros(g.out + h.out, gin)
        out(0 -> g.out, ->) = g.∇(x)
        out
      case (0, hin) =>
        val out = zeros(g.out + h.out, hin)
        out(g.out ->, ->) = h.∇(x)
        out
      case (gin, hin) =>
        val xg = x(0 -> gin)
        val xh = x(gin ->)
        val out = zeros(g.out + h.out, gin + hin)
        out(0 -> g.out, 0 -> gin) = g.∇(xg)
        out(g.out ->, gin ->) = h.∇(xh)
        out
    }
  }

  override def toString(): String = s"Concat-$in-$out($g, $h)"
}

object Concat {

  def apply(f: Differentiable*): Differentiable =
    f.length match {
      case 1 => f(0)
      case n =>
        val (a, b) = f.splitAt(n / 2)
        Concat(a: _*) ++ Concat(b: _*)
    }
}