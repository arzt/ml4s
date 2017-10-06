package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.abs
import org.nd4s.Implicits._

import scala.Iterator.iterate

trait Differentiable extends ((ND) => ND) {
  def in: Int

  def out: Int

  def rows: Int = 1

  def cols: Int = out

  def check(x: ND): Unit = {
    assert(x.length() == in, s"Dimensions are not equal: ${x.length()} != $in")
  }

  def apply(x: ND): ND

  def ∇(x: ND): ND

  def unary_∇ : ND => ND = x => ∇(x)

  def *(h: Differentiable): Differentiable = Compose(this, h)

  def ∘(h: Differentiable): Differentiable = Compose(this, h)

  //def **(h: Differentiable): Differentiable = Compose(this, h)

  def ++(h: Differentiable): Differentiable = new Concat(this, h)

  //def +(h: Differentiable): Differentiable = new Concat(this, h)

  def empiricalGrad(x: ND, ε: Double): ND = {
    if (in == 0) {
      null
    } else {
      val ∇ = Nd4j.zeros(out, in)
      (0 until x.length())
        .foreach { i =>
          val t = x(i)
          x(i) = t + ε
          val la = this (x.dup())
          x(i) = t - ε
          val lb = this (x.dup())
          x(i) = t
          val d = (la - lb) / 2 / ε
          ∇(->, i) = d
        }
      ∇
    }
  }


  def gradientDescend(x: ND, α: Double): Iterator[Step] = {
    assert(out == 1, s"Function must have one output dimension, got $out")
    val m = -1
    val y = apply(x).getDouble(0)
    val init = Step(x, ∇(x), m)
    iterate(init) { case Step(x, oldGrad, _) =>
      val grad = this.∇(x)
      x(->) = x - grad * α
      val gradDiff = abs(grad - oldGrad).maxNumber().doubleValue()
      Step(x, grad, gradDiff)
    }
  }

  def maxDelta(x: ND, y: ND): Double = {
    val mY = apply(x)
    abs(mY - y)
      .maxNumber()
      .doubleValue()
  }

  def maxGradDelta(x: ND, ε: Double): Double = {
    val b = empiricalGrad(x, ε)
    val a = ∇(x)
    assert(b.rows() == a.rows() && b.columns() == a.columns(), "shapes must be equal")
    val δ = abs(a - b)
    δ.maxNumber().doubleValue()
  }

}
