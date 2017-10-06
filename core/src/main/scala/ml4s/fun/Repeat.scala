package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{concat, diag, ones}

class Repeat(val in: Int, val times: Int) extends Differentiable {
  val out: Int = in * times

  override def apply(x: ND): ND = {
    require(x.length() == in, "input dimension must match.")
    concat(1, Iterator.fill(times)(x).toSeq: _*)
  }

  override def ∇(x: ND): ND = {
    val d = diag(ones(in))
    concat(0, Iterator.fill(times)(d).toSeq: _*)
  }

  override def toString(): String = s"Repeat-$in-$out"
}

object Repeat {
  def apply(in: Int, times: Int): Repeat = new Repeat(in, times)

  def apply(f: Differentiable, times: Int): Differentiable = Repeat(f.out, times) ∘ f

  def apply(f: Differentiable, g: Differentiable): Differentiable = (f ++ g) ∘ Repeat(f.in, 2)
}