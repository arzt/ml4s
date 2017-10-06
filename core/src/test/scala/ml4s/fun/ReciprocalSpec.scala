package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.zeros
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class ReciprocalSpec extends Specification {
  "Reciprocal Should" should {
    val zero = zeros(1, 3)
    "compute reciprocal" in {
      val rec = Reciprocal(3)
      val δ = rec.maxDelta(zero + 2, zero + 0.5)
      δ must beLessThan(0.01)
    }
    "compute derivative" in {
      val rec = Reciprocal(3)
      val x = zero + 2
      val δ = rec.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }
}
