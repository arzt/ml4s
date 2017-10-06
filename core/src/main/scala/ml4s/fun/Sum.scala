package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j

class Sum(val out: Int, val addends: Int) extends Differentiable {
  val in: Int = out * addends

  override def apply(x: ND): ND = {
    val res = x.reshape(addends, out)
    val y = res.sum(0)
    y.ravel()
  }

  override def ∇(x: ND): ND = {
    val r =
      (0 until addends)
        .view
        .map { i =>
          Nd4j.diag(Nd4j.ones(1, out))
        }
    Nd4j.concat(1, r: _*)
  }
}


object Sum {

  def apply(in: Int): Sum = new Sum(1, in)

  def apply(f: Differentiable): Differentiable = Sum(1, f.out) * f

  def apply(out: Int, summands: Int): Sum = new Sum(out, summands)

  def apply(f: Differentiable*): Differentiable = {
    val n = f.length
    Sum(f(0).out, n) * Concat(f: _*) * Repeat(f(0).in, n)
  }

  def apply(f: Differentiable, g: Differentiable): Differentiable = {
    val h = f ++ g
    Sum(f.out, 2) * h
  }

  def apply(f: Differentiable, v: Double): Differentiable = {
    new Sum(f.out, 2) * (f ++ Const(0, f.out, v))
  }
}