package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j.{concat, randn}
import org.specs2.mutable.Specification

class ConcatSpec extends Specification {
  val ε = 0.01
  "Concat" should {
    val h = Id(4)
    val g = Id(5)
    "concat 2 functions" in {
      val xg = randn(5, 1)
      val xh = randn(4, 1)
      val x = concat(0, xh, xg)
      val y = concat(0, h(xh), g(xg))
      val δ = Concat(h, g).maxDelta(x, y)
      δ must beCloseTo(0, ε)
    }
    "compute gradient" in {
      val xg = randn(5, 1)
      val xh = randn(4, 1)
      val x = concat(0, xh, xg)
      val δ = Concat(h, g).maxGradDelta(x, ε)
      δ must beCloseTo(0, ε)
    }
    "work with zero dimensional functions" in {
      val v = randn(5, 1)
      val f = Const(0, v)
      val stack = Concat(h, f)
      val δ = stack.maxGradDelta(randn(4, 1), ε)
      δ must beCloseTo(0, ε)
    }
    "work with zero dimensional functions2" in {
      val v = randn(5, 1)
      val f = Const(0, v)
      val stack = Concat(f, h)
      val δ = stack.maxGradDelta(randn(4, 1), ε)
      δ must beCloseTo(0, ε)
    }
    "work with zero dimensional functions" in {
      val v = randn(5, 1)
      val f = Const(0, v)
      val xg = randn(5, 1)
      val xh = randn(4, 1)
      val x = concat(0, xh, xg)
      Concat(f, f).∇(x) must beNull
    }
    "concat multiple functions" in {
      val fa = Id(3)
      val xa = randn(1, 3)
      val ya = fa(xa)

      val fb = Exp(4)
      val xb = randn(1, 4)
      val yb = fb(xb)

      val fc = Log(5)
      val xc = randn(1, 5)
      val yc = fc(xc)

      val fd = Square(6)
      val xd = randn(1, 6)
      val yd = fd(xd)

      val fe = Max(4, 0)
      val xe = randn(1, 4)
      val ye = fe(xe)

      val x = concat(1, xa, xb, xc, xd, xe)
      val con = Concat(fa, fb, fc, fd, fe)
      val y = concat(1, ya, yb, yc, yd, ye)
      val δ = con.maxDelta(x, y)
      δ must beLessThan(0.01)
    }
  }

}
