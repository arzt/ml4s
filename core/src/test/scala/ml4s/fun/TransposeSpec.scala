package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j._
import org.specs2.mutable.Specification

class TransposeSpec extends Specification {
  "Transpose" should {
    "compute the transpose" in {
      val X = randn(2, 3)
      val x = X.ravel()
      val y = X.transpose().ravel()
      val t = new Transpose(2, 3)
      val δ = t.maxDelta(x, y)
      δ should beLessThan(0.01)
    }
    "compute derivative" in {
      val m = 7
      val n = 5
      val X = randn(m, n)
      val x = X.ravel()
      val t = new Transpose(m, n)
      val δ = t.maxGradDelta(x, 0.01)
      δ should beLessThan(0.01)
    }
  }
}
