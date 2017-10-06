package ml4s
package fun

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.{create, diag, zeros,valueArrayOf}
import org.nd4s.Implicits._

import scala.language.postfixOps

class MatProd(m: Int, l: Int, n: Int) extends Differentiable {
  val in: Int = m * l + l * n
  val out: Int = m * n

  override def apply(x: ND): ND = {
    assert(x.length() == in)
    val A = x(0 -> m * l).reshape(m, l)
    val B = x(l * m ->).reshape(l, n)
    (A ** B).ravel()
  }

  override def ∇(x: ND): ND = {
    assert(x.length() == in)
    val lm = l * m
    val ln = l * n
    val A = x(0 -> lm).reshape(m, l)
    val B = x(lm ->).reshape(l, n)
    val g = zeros(out, lm + ln)

    (0 until m)
      .foreach { i =>
        val sr = i * n
        val sc = i * l
        val rows = sr -> (sr + n)
        val cols = sc -> (sc + l)
        g(rows, cols) = B.transpose()
        (0 until l)
          .foreach { j =>
            val cols2 = (lm + j * n) -> (lm + j * n + n)
            g(rows, cols2) = diag(valueArrayOf(n, A(i, j)))
          }
      }
    g
  }

  def old(x: ND): ND = {
    assert(x.length() == in)
    val lm = l * m
    val ln = l * n
    val A = x(0 -> lm).reshape(m, l)
    val B = x(lm ->).reshape(l, n)
    val g = zeros(out, lm + ln)
    (0 until m)
      .foreach { i =>
        val sr = i * n
        val sc = i * l
        val rows = sr -> (sr + n)
        val cols = sc -> (sc + l)
        g(rows, cols) = B.transpose()
        (0 until l)
          .foreach { col =>
            val cols2 = (lm + col * n) -> (lm + col * n + n)
            g(rows, cols2) = diag(zeros(n) + A(i, col))
          }
      }
    g
  }
}

object MatProd {
  def apply(m: Int, l: Int, n: Int): MatProd = new MatProd(m, l, n)
}