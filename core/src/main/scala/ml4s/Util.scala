package ml4s

import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

object Util {

  def toBinaryMax(x: ND): ND = {
    val argMax = x.argMax(0)
    abs(x, false)
    x *= 0
    (0 until x.columns())
      .foreach { c =>
        x(argMax(c).toInt, c) = 1
      }
    x
  }

}
