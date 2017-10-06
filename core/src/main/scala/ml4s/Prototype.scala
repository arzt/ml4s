package ml4s

import ml4s.fun._
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

object Prototype {

  def main(args: Array[String]): Unit = {
    val dimIn = 10
    val N = 10
    val dimOut = 5
    val x = Nd4j.rand(dimIn, N)
    val w = Nd4j.randn(dimOut, dimIn).ravel()
    val w2 = Nd4j.randn(dimOut, dimIn).ravel()
    val size = dimOut * dimIn

    val perc = MatProd(N, dimIn, dimOut) ∘ (Id(size) ++ Const(0, x))
    val sig = Sigmoid(perc)
    val y = sig(w)
    val y2 = y.reshape(dimOut, N)
    val label = Const(sig.in, y)
    val fun = Mean(HalfSquare(Minus(sig, label)))
    fun
      .gradientDescend(w2, 2)
      .foreach(println)
  }

  def nd4jBug(args: Array[String]): Unit = {
    val data = Array[Double](1, 2, 3,
                             4, 5, 6,
                             7, 8, 9)
    val mat = Nd4j.create(data, Array(3, 3))
    val stride = 3
    val rows = NDArrayIndex.interval(1, stride, 2)
    val cols = NDArrayIndex.interval(1, stride, 2)
    val result = mat.get(rows, cols)
    val y1 = result.getDouble(0)
    val y2 = mat.getDouble(1, 1)
    assert(y1 == y2, "Must be equal")
  }
}
