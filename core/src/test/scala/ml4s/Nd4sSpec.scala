package ml4s

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class Nd4sSpec extends Specification {

  "eps" should {
    "compute eps comparison" in {
      val x = Nd4j.zeros(1, 10) + 1
      val x2 = Nd4j.zeros(1, 10) + 1.002
      1 === 1
    }
  }
  "agrMax" should {
    "work" in {
      val x = Nd4j.randn(10, 3)
      val argMax = x.argMax(0)
      val amax = x.amax(0)
      val max = x.max(0)
      1 === 1
    }
  }
  "squaredDistance" should {
    "compute scared distance" in {
      val data = Array[Float](1, 2, 3,
                              4, 5, 6,
                              7, 8, 9)
      val x = Nd4j.create(data, Array(3, 3))
      val x2 = x.dup()
      x2(0, 0) = 3
      val res = x.squaredDistance(x2)
      res must beCloseTo(4d, 0.0001)
    }
  }
  "eq(0)" should {
    "inverse binary matrix" in {
      val bin = Nd4j.randn(3, 3).lte(0)
      val inverse = bin.eq(0)
      val sum = bin + inverse
      sum.maxNumber().doubleValue() must beCloseTo(1, 0.001)
      sum.minNumber().doubleValue() must beCloseTo(1, 0.001)
    }
  }
  "slicing" should {
    "worko" in {
      val x = Nd4j.randn(3, 3)
      val rows = 1 -> 2 by 3
      val cols = 1 -> 2 by 3
      val y = x(rows, cols)
      val y2 = x(rows, ->).apply(->, cols)
      1 === 1
    }
  }

}
