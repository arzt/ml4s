package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

class Sigmoid(val in: Int) extends Differentiable {
  val out: Int = in

  override def apply(x: ND): ND = sigmoid(x)

  override def ∇(x: ND): ND = {
    val sigm = sigmoid(x)
    diag(sigm * (-sigm + 1))
  }
}

object Sigmoid {
  def apply(in: Int): Sigmoid = new Sigmoid(in)

  def apply(f: Differentiable): Differentiable = Compose(Sigmoid(f.out), f)
}