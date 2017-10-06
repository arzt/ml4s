package ml4s

import ml4s.fun._
import org.nd4j.linalg.factory.Nd4j._
import org.specs2.mutable.Specification

class FunSpec extends Specification {
  "Nabla" should {
    "compute mean square error" in {
      val dimIn = 10
      val N = 10
      val dimOut = 5
      val x = rand(dimIn, N)
      val w2 = randn(dimOut, dimIn).ravel()
      val size = dimOut * dimIn
      val perc = MatProd(N, dimIn, dimOut) ∘ (Id(size) ++ Const(0, x))
      val w = randn(1, perc.in).ravel()
      val y = perc(w)
      val label = Const(0, y)
      val fun = MeanSquaredError(perc, label)
      val δ = fun.maxGradDelta(w2, 0.01)
      δ must beLessThan(0.01)
    }
  }
}
