package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.{randn, zeros}
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class PowSpec extends Specification {
  val ε = 0.01

  "Pow" should {
    "compute elementwise power" in {
      val zero = zeros(1, 3)
      val exponents = zero - 1
      val pow = Pow(exponents)
      val x = zero + 2
      val y = zero + 0.5
      val δ = pow.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute elementwise power2" in {
      val zero = zeros(1, 3)
      val exponents = zero + 2
      val pow = Pow(exponents)
      val x = zero.dup()
      x(0) = 1
      x(1) = 2
      x(2) = 3
      val y = zero.dup()
      y(0) = 1
      y(1) = 4
      y(2) = 9
      val δ = pow.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
    "compute gradient" in {
      val zero = zeros(1, 3)
      val exponents = zero + 2
      val pow = Pow(exponents)
      val x = zero.dup()
      x(0) = 1
      x(1) = 2
      x(2) = 3
      val y = zero.dup()
      y(0) = 1
      y(1) = 4
      y(2) = 9
      val δ = pow.maxGradDelta(x, ε)
      δ must beCloseTo(0, ε)
    }
  }
}
