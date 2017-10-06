package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{diag, zeros}
import org.nd4s.Implicits._

import scala.math.pow

class Pow(exponents: ND) extends Differentiable {
  val in: Int = exponents.length
  val out: Int = in

  override def apply(x: ND): ND = {
    assert(x.length() == in)
    (0 until in)
      .foreach { i =>
        x(i) = pow(x(i), exponents(i))
      }
    x
  }

  override def ∇(x: ND): ND = {
    val scale = Scale(exponents, Pow(exponents - 1))
    val hui = Pow(exponents - 1)
    val kh = hui(x.dup())
    val res = scale(x.dup())
    diag(res)
  }

  override def toString: String = "x^n"
}

object Pow {
  def apply(exponents: ND): Pow = new Pow(exponents)

  def apply(in: Int, exponent: Double): Pow = Pow(zeros(1, in) + exponent)

  def apply(f: Differentiable, exponents: ND): Differentiable = Pow(exponents) ∘ f

  def apply(f: Differentiable, exponent: Double): Differentiable = Pow(f.out, exponent) ∘ f
}