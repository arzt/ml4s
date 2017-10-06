package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{concat, diag}
import org.nd4s.Implicits._

class Prod(val in: Int) extends Differentiable {
  require(in % 2 == 0)
  val out: Int = in / 2

  override def apply(x: ND): ND = {
    val a = x(0 -> out)
    val b = x(out ->)
    a * b
  }

  override def ∇(x: ND): ND = {
    val a = x(0 -> out)
    val b = x(out ->)
    val db = diag(b)
    val da = diag(a)
    concat(1, db, da)
  }

  override def toString(): String = s"Prod-$in-$out"
}

object Prod {
  def apply(in: Int): Prod = new Prod(in)

  def apply(f: Differentiable): Differentiable = Prod(f.out) * f

  def apply(f: Differentiable, g: Differentiable): Differentiable = {
    require(f.out == g.out, "output dimensions must be equal")
    Prod(f ++ g)
  }
}