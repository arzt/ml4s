package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

class Log(val in: Int) extends Differentiable {
  val out: Int = in

  override def apply(x: ND): ND = log(x)

  override def ∇(x: ND): ND = diag(pow(x, -1))

}

object Log {
  def apply(in: Int): Log = new Log(in)

  def apply(f: Differentiable): Differentiable = Log(f.out) * f
}