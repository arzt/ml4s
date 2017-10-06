package ml4s

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class UtilSpec extends Specification {
  "toLabel" should {
    "convert probabilities to label" in {
      val data = Array[Double](7, 2, 2,
                               3, 0, 1,
                               5, 4, 9)
      val x = Nd4j.create(data, Array(3, 3))
      val y = Nd4j.zeros(3, 3)
      y(0, 0) = 1
      y(2, 1) = 1
      y(2, 2) = 1
      Util.toBinaryMax(x)
      val δ = (x - y).maxNumber().doubleValue()
      δ must beCloseTo(0, 0.01)
    }
  }
}
