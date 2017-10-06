package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.rand
import org.specs2.mutable.Specification

class MapRowsSpec extends Specification {
  "MapRows" should {
    "map rows of a matrix" in {
      val id = Id(4)
      val map = MapRows(4, id)
      val x = rand(1, 16)
      val y = x.dup()
      val δ = map.maxDelta(x, y)
      δ must beLessThan(0.001)
    }
    "comput derivative" in {
      val id = Sum(4)
      val map = MapRows(4, id)
      val x = rand(1, 16)
      val δ = map.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }
}
