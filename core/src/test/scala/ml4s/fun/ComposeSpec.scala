package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.randn
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class ComposeSpec extends Specification {
  lazy val ε = 0.01
  "Compose" should {
    "chain functions" in {
      val a = Id(4)
      val b = Id(4)
      val chain = Compose(b, a)
      val x = randn(1, 4)
      val y = x
      val δ = chain.maxDelta(x, y)
      δ must beCloseTo(0, ε)
    }
    "compute derivative" in {
      val a = Id(4)
      val b = Id(4)
      val chain = Compose(b, a)
      val x = randn(1, 4)
      val δ = chain.maxGradDelta(x, ε)
      δ must beCloseTo(0, ε)
    }
  }
}
