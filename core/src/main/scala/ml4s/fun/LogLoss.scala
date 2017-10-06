package ml4s
package fun

object LogLoss {
  def apply(in: Int, label: ND): Differentiable = {
    val ε = 1e-15
    val zeros = Const(0, label lt 0.5)
    val ones = Const(0, label gt 0.5)
    val np = Sum(Neg(in), 1)
    val logp = Log(Max(in, ε))
    val lognp = Log(Max(np, ε))
    Neg(Sum(Prod(ones, logp), Prod(zeros, lognp)) * Repeat(in, 2))
  }
}
