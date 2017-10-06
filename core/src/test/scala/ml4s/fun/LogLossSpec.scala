package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms.{log, max, softmax}
import org.nd4s.Implicits._
import org.specs2.mutable.Specification

class LogLossSpec extends Specification {

  def logLoss(x: ND, label: ND): ND = {
    val ε: Double = 1e-15
    val zeros = label lt 0.5
    val ones = label gt 0.5
    val p = x.dup()
    val np = p * -1 + 1
    max(p, ε, false) //to avoid minus infinity, when taking log
    max(np, ε, false)
    val logp = log(p)
    val lognp = log(np)
    val a = ones * logp
    val b = zeros * lognp
    -(a + b)
  }

  "Log loss" should {
    "compute log loss" in {
      val label = randn(4, 5)
      val y = softmax(label.transpose(), true).transpose()
      Util.toBinaryMax(label)
      val f = LogLoss(20, label.ravel())
      val loss1 = f(y.ravel)
      val loss2 = logLoss(y.ravel(), label.ravel())
      val res1 = loss1.sumNumber().doubleValue()
      val res2 = loss2.sumNumber().doubleValue()
      res1 must beCloseTo(res2, 0.0001)
    }
    "compute log loss" in {
      val label = randn(4, 5)
      val y = softmax(label.transpose(), true).transpose()
      Util.toBinaryMax(label)
      val loss = LogLoss(20, label.ravel())
      val lo = loss(label.ravel())
      val lo2 = loss(y.ravel)
      val res = lo.sumNumber().doubleValue()
      val res2 = lo2.sumNumber().doubleValue()
      res must beCloseTo(0, 00.1)
      res2 must beGreaterThan(res)
    }
    "compute derivative" in {
      val label = randn(4, 5)
      val y = softmax(label.transpose(), true).transpose()
      Util.toBinaryMax(label)
      val loss = LogLoss(20, label.ravel())
      val x = y.ravel()
      val δ = loss.maxGradDelta(x, 0.01)
      δ must beLessThan(0.01)
    }
  }
}
